from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import anyio
import pytest
from deckr.contracts.lanes import CORE_LANE_CONTRACTS, LaneContractRegistry
from deckr.contracts.messages import (
    EndpointAddress,
    controller_address,
    hardware_manager_address,
)
from deckr.hardware import messages as hw_messages
from deckr.hardware.descriptors import (
    DECKR_INPUT_BUTTON,
    DECKR_OUTPUT_RASTER,
    CapabilityDescriptor,
    CapabilityRef,
    ControlDescriptor,
    ControlGeometry,
    DeviceDescriptor,
    DeviceRef,
)
from deckr.lanes import RegisteredEndpointLane
from deckr.runtime import Deckr
from deckr.state import (
    DEFAULT_DISCOVERY_STATE_STORE_NAME,
    DEFAULT_LEASE_STATE_STORE_NAME,
    DeviceClaim,
    EndpointPresence,
    HardwareInventory,
    StateUnavailable,
    hardware_inventory_key,
    presence_endpoint_key,
)
from memory_lane_substrate import MemoryLaneSubstrate

from deckr.drivers.elgato._device import ElgatoDockDevice
from deckr.drivers.elgato._discovery import _apply_device_commands, discover_loop
from deckr.drivers.elgato._factory import ElgatoDeviceFactory, driver_factory

MANAGER_SESSION = "manager-session"
CONTROLLER_SESSION = "controller-session"


class EndpointHarness:
    def __init__(
        self,
        deckr: Deckr,
        endpoint: EndpointAddress,
        *,
        session_id: str,
    ) -> None:
        self._state = deckr.state()
        self._registered = RegisteredEndpointLane(
            lane=deckr.lane("hardware_messages"),
            endpoint=endpoint,
            session_id=session_id,
            state=self._state,
            metadata={"runtime": "test"},
        )
        self._presence_revision: int | None = None

    @property
    def lane(self):
        return self._registered.lane

    @property
    def endpoint(self) -> EndpointAddress:
        return self._registered.endpoint

    @property
    def session_id(self) -> str:
        return self._registered.session_id

    async def _ensure_presence(self) -> None:
        entry = await self._state.put(
            presence_endpoint_key(lane=self.lane.name, endpoint=self.endpoint),
            EndpointPresence(
                endpoint=self.endpoint,
                lane=self.lane.name,
                sessionId=self.session_id,
                timestamp=datetime.now(UTC),
                ttlSeconds=30,
            ),
            ttl=30,
        )
        self._presence_revision = entry.revision

    async def publish(self, message):
        await self._ensure_presence()
        return await self._registered.publish(message)

    @asynccontextmanager
    async def subscribe(self) -> AsyncIterator:
        await self._ensure_presence()
        async with self._registered.subscribe() as stream:
            yield stream

    async def __aenter__(self) -> EndpointHarness:
        await self._ensure_presence()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        key = presence_endpoint_key(lane=self.lane.name, endpoint=self.endpoint)
        revision = self._presence_revision
        if revision is not None:
            await self._state.delete(key, revision=revision)
        self._presence_revision = None


def _endpoint(
    deckr: Deckr,
    endpoint: EndpointAddress,
    *,
    session_id: str = CONTROLLER_SESSION,
) -> EndpointHarness:
    return EndpointHarness(deckr, endpoint, session_id=session_id)


class _FakeDeviceManager:
    def __init__(self, devices: list[object]) -> None:
        self._devices = devices

    def enumerate(self) -> list[object]:
        return self._devices


class _FakeStreamDeck:
    def key_layout(self) -> tuple[int, int]:
        return 1, 1


def _deckr() -> Deckr:
    lane_contracts = LaneContractRegistry(CORE_LANE_CONTRACTS.values())
    return Deckr(
        lane_contracts=lane_contracts,
        substrate=MemoryLaneSubstrate(lane_contracts=lane_contracts),
    )


def _control() -> ControlDescriptor:
    return ControlDescriptor(
        controlId="0,0",
        kind="bitmap_key",
        geometry=ControlGeometry(x=0, y=0, width=1, height=1, unit="grid"),
        inputCapabilities=(
            CapabilityDescriptor(
                capabilityId="button.momentary",
                family=DECKR_INPUT_BUTTON,
                type="momentary",
                direction="input",
                access=("emits",),
                eventTypes=("down", "up"),
            ),
        ),
        outputCapabilities=(
            CapabilityDescriptor.model_validate(
                {
                    "capabilityId": "raster.bitmap",
                    "family": DECKR_OUTPUT_RASTER,
                    "type": "bitmap",
                    "direction": "output",
                    "access": ["settable"],
                    "commandTypes": ["set_frame", "clear"],
                    "constraints": [
                        {"type": "fixed", "subject": "width", "value": 72},
                        {"type": "fixed", "subject": "height", "value": 72},
                    ],
                }
            ),
        ),
    )


def _device() -> DeviceDescriptor:
    return DeviceDescriptor(
        deviceId="deck",
        displayName="Stream Deck",
        fingerprint="fingerprint:deck",
        controls=(_control(),),
    )


def _available_message() -> hw_messages.DeviceAvailableMessage:
    return hw_messages.device_available_message(
        manager_id="elgato-main",
        sender_session_id=MANAGER_SESSION,
        descriptor=_device(),
    )


def _unavailable_message() -> hw_messages.DeviceUnavailableMessage:
    return hw_messages.device_unavailable_message(
        manager_id="elgato-main",
        sender_session_id=MANAGER_SESSION,
        device_id="deck",
        reason="test",
    )


def _input_message() -> hw_messages.ControlInputMessage:
    return hw_messages.control_input_message(
        manager_id="elgato-main",
        sender_session_id=MANAGER_SESSION,
        device_id="deck",
        control_id="0,0",
        capability_id="button.momentary",
        event_type="down",
        value={"eventType": "down"},
    )


def _command_message(controller_id: str, image: bytes) -> hw_messages.ControlCommandMessage:
    return hw_messages.control_command_for_capability(
        controller_id=controller_id,
        sender_session_id=CONTROLLER_SESSION,
        ref=CapabilityRef(
            deviceRef=DeviceRef(managerId="elgato-main", deviceId="deck"),
            controlId="0,0",
            capabilityId="raster.bitmap",
        ),
        command_type="set_frame",
        params={
            "image": base64.b64encode(image).decode("ascii"),
            "encoding": "jpeg",
        },
    )


def _power_command_message(controller_id: str, command_type: str) -> hw_messages.ControlCommandMessage:
    return hw_messages.control_command_for_capability(
        controller_id=controller_id,
        sender_session_id=CONTROLLER_SESSION,
        ref=CapabilityRef(
            deviceRef=DeviceRef(managerId="elgato-main", deviceId="deck"),
            capabilityId="device.power",
        ),
        command_type=command_type,
        params={},
    )


def _factory(deckr: Deckr) -> ElgatoDeviceFactory:
    manager = ElgatoDeviceFactory(
        deckr.lane("hardware_messages"),
        deckr.state(DEFAULT_LEASE_STATE_STORE_NAME),
        deckr.state(DEFAULT_DISCOVERY_STATE_STORE_NAME),
        manager_id="elgato-main",
    )
    manager._endpoint = _endpoint(
        deckr,
        hardware_manager_address("elgato-main"),
        session_id=MANAGER_SESSION,
    )
    manager._endpoint_cm = manager._endpoint
    manager._session_id = manager._endpoint.session_id
    return manager


def _claim(controller_id: str = "main", session_id: str = "controller-session"):
    return DeviceClaim(
        claimedByEndpoint=controller_address(controller_id),
        claimedBySessionId=session_id,
        timestamp=datetime.now(UTC),
        ttlSeconds=30,
    )


async def _put_controller_presence(
    deckr: Deckr,
    *,
    controller_id: str = "main",
    session_id: str = "controller-session",
) -> None:
    endpoint = controller_address(controller_id)
    await deckr.state(DEFAULT_LEASE_STATE_STORE_NAME).put(
        presence_endpoint_key(lane="hardware_messages", endpoint=endpoint),
        EndpointPresence(
            endpoint=endpoint,
            lane="hardware_messages",
            sessionId=session_id,
            timestamp=datetime.now(UTC),
            ttlSeconds=30,
            metadata={},
        ),
    )


def test_elgato_descriptor_exposes_only_momentary_button_input() -> None:
    device = ElgatoDockDevice(_FakeStreamDeck())

    controls = device._create_controls()

    assert len(controls) == 1
    assert [
        (capability.capability_id, capability.event_types)
        for capability in controls[0].input_capabilities
    ] == [("button.momentary", ("down", "up"))]


@pytest.mark.asyncio
async def test_elgato_key_up_emits_only_momentary_up() -> None:
    device = ElgatoDockDevice(_FakeStreamDeck())

    await device._on_key_event(object(), 0, False)

    with anyio.fail_after(1):
        event = await device._event_receive.receive()
    assert event.control_id == "0,0"
    assert event.capability_id == "button.momentary"
    assert event.event_type == "up"
    with anyio.move_on_after(0.05) as scope:
        await device._event_receive.receive()
    assert scope.cancel_called is True


def test_driver_factory_returns_elgato_device_factory() -> None:
    async def check() -> None:
        async with _deckr() as deckr:
            factory = driver_factory(
                deckr.lane("hardware_messages"),
                deckr.state(DEFAULT_LEASE_STATE_STORE_NAME),
                deckr.state(DEFAULT_DISCOVERY_STATE_STORE_NAME),
                manager_id="elgato-main",
            )
            assert isinstance(factory, ElgatoDeviceFactory)

    anyio.run(check)


@pytest.mark.asyncio
async def test_discover_loop_emits_first_available_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_device = object()
    monkeypatch.setattr(
        "deckr.drivers.elgato._discovery.DeviceManager",
        lambda: _FakeDeviceManager([fake_device]),
    )
    send_stream, receive_stream = anyio.create_memory_object_stream[object](
        max_buffer_size=1
    )
    device_connected = [False]

    async with anyio.create_task_group() as tg:
        tg.start_soon(discover_loop, send_stream, device_connected)

        with anyio.fail_after(2):
            assert await receive_stream.receive() is fake_device

        device_connected[0] = True
        with anyio.move_on_after(0.2) as scope:
            await receive_stream.receive()
        assert scope.cancel_called is True

        tg.cancel_scope.cancel()


@pytest.mark.asyncio
async def test_connect_and_disconnect_rewrite_aggregate_inventory() -> None:
    async with _deckr() as deckr:
        manager = _factory(deckr)
        await manager._handle_device_message(
            _available_message()
        )
        entry = await deckr.state(DEFAULT_DISCOVERY_STATE_STORE_NAME).get(
            hardware_inventory_key("elgato-main")
        )
        assert entry is not None
        inventory = HardwareInventory.model_validate(entry.value)
        assert set(inventory.devices) == {"deck"}
        assert inventory.devices["deck"].descriptor.device_id == "deck"

        await manager._handle_device_message(
            _unavailable_message()
        )
        entry = await deckr.state(DEFAULT_DISCOVERY_STATE_STORE_NAME).get(
            hardware_inventory_key("elgato-main")
        )
        assert entry is not None
        inventory = HardwareInventory.model_validate(entry.value)
        assert inventory.devices == {}


@pytest.mark.asyncio
async def test_inventory_state_unavailable_keeps_local_device_state() -> None:
    class UnavailableState:
        async def put(self, *args, **kwargs):
            raise StateUnavailable("temporary substrate outage")

    async with _deckr() as deckr:
        manager = ElgatoDeviceFactory(
            deckr.lane("hardware_messages"),
            deckr.state(DEFAULT_LEASE_STATE_STORE_NAME),
            UnavailableState(),
            manager_id="elgato-main",
        )
        manager._endpoint = _endpoint(
            deckr,
            hardware_manager_address("elgato-main"),
            session_id=MANAGER_SESSION,
        )

        await manager._handle_device_message(
            _available_message()
        )

    assert "deck" in manager._devices
    assert manager._inventory_revision is None


@pytest.mark.asyncio
async def test_inventory_publish_writes_aggregate_inventory() -> None:
    async with _deckr() as deckr:
        manager = _factory(deckr)
        manager._devices["deck"] = _device()

        await manager._publish_inventory_safely()
        entry = await deckr.state(DEFAULT_DISCOVERY_STATE_STORE_NAME).get(
            hardware_inventory_key("elgato-main")
        )
        assert entry is not None

    inventory = HardwareInventory.model_validate(entry.value)
    assert set(inventory.devices) == {"deck"}


@pytest.mark.asyncio
async def test_claimed_input_is_sent_only_to_claiming_controller() -> None:
    async with _deckr() as deckr:
        manager = _factory(deckr)
        manager._devices["deck"] = _device()
        manager._claims["deck"] = _claim()
        manager._controller_presence_sessions[controller_address("main")] = (
            "controller-session"
        )
        main = _endpoint(deckr, controller_address("main"))
        other = _endpoint(deckr, controller_address("other"))

        async with main.subscribe() as main_stream, other.subscribe() as other_stream:
            await manager._handle_device_message(
                _input_message()
            )
            received = await main_stream.receive()
            with anyio.move_on_after(0.05) as scope:
                await other_stream.receive()

    assert received.recipient.endpoint == controller_address("main")
    assert scope.cancel_called


@pytest.mark.asyncio
async def test_broker_snapshot_claim_delete_resets_device_and_drops_input() -> None:
    class FakeDevice:
        id = "deck"

        def __init__(self) -> None:
            self.clear_key = AsyncMock()
            self.refresh = AsyncMock()

    async with _deckr() as deckr:
        manager = _factory(deckr)
        manager._devices["deck"] = _device()
        device = FakeDevice()
        command_send, command_receive = anyio.create_memory_object_stream(max_buffer_size=100)
        manager._command_streams["deck"] = command_send
        await _put_controller_presence(deckr)
        claim_key = "claim.device.elgato-main.deck"
        await deckr.state().create(claim_key, _claim())
        await manager._reconcile_routing_current_state(reason="test snapshot")
        main = _endpoint(deckr, controller_address("main"))

        async with (
            command_send,
            command_receive,
            main.subscribe() as main_stream,
            anyio.create_task_group() as tg,
        ):
            tg.start_soon(
                _apply_device_commands,
                device,
                command_receive,
                "elgato-main",
            )
            await deckr.state().delete(claim_key)
            await manager._reconcile_routing_current_state(reason="test snapshot")
            with anyio.fail_after(1):
                while device.clear_key.await_count < 1:
                    await anyio.sleep(0.01)

            await manager._handle_device_message(
                _input_message()
            )
            with anyio.move_on_after(0.05) as scope:
                await main_stream.receive()
            tg.cancel_scope.cancel()

    device.clear_key.assert_awaited_once()
    device.refresh.assert_awaited_once()
    assert scope.cancel_called


@pytest.mark.asyncio
async def test_prefix_observation_omissions_keep_current_routing(monkeypatch) -> None:
    async with _deckr() as deckr:
        manager = _factory(deckr)
        manager._devices["deck"] = _device()
        await _put_controller_presence(deckr)
        await deckr.state().create("claim.device.elgato-main.deck", _claim())
        await manager._reconcile_routing_current_state(reason="initial snapshot")
        assert manager._claim_recipient("deck") == controller_address("main")

        async def omitted_items(prefix: str = ""):
            del prefix
            return ()

        monkeypatch.setattr(
            deckr.state(DEFAULT_LEASE_STATE_STORE_NAME),
            "items",
            omitted_items,
        )

        await manager._reconcile_routing_current_state(reason="omitted snapshot")

        assert manager._claim_recipient("deck") == controller_address("main")
        assert "deck" in manager._claims


@pytest.mark.asyncio
async def test_controller_presence_restore_makes_current_claim_routable() -> None:
    async with _deckr() as deckr:
        manager = _factory(deckr)
        manager._devices["deck"] = _device()
        claim_key = "claim.device.elgato-main.deck"
        await deckr.state().create(claim_key, _claim())
        await manager._reconcile_routing_current_state(reason="test snapshot")
        assert manager._claim_recipient("deck") is None

        await _put_controller_presence(deckr)
        await manager._reconcile_routing_current_state(reason="test snapshot")
        assert manager._claim_recipient("deck") == controller_address("main")

        main = _endpoint(deckr, controller_address("main"))
        async with main.subscribe() as main_stream:
            await manager._handle_device_message(
                _input_message()
            )
            received = await main_stream.receive()

    assert received.recipient.endpoint == controller_address("main")


@pytest.mark.asyncio
async def test_invalid_claim_payload_is_not_routable() -> None:
    class FakeDevice:
        id = "deck"

        def __init__(self) -> None:
            self.clear_key = AsyncMock()
            self.refresh = AsyncMock()

    async with _deckr() as deckr:
        manager = _factory(deckr)
        manager._devices["deck"] = _device()
        device = FakeDevice()
        command_send, command_receive = anyio.create_memory_object_stream(max_buffer_size=100)
        manager._command_streams["deck"] = command_send
        await deckr.state().put(
            "claim.device.elgato-main.deck",
            {
                "claimedByEndpoint": "controller:main",
                "timestamp": datetime.now(UTC).isoformat(),
                "ttlSeconds": 30,
            },
        )
        await _put_controller_presence(deckr)

        async with command_send, command_receive, anyio.create_task_group() as tg:
            tg.start_soon(
                _apply_device_commands,
                device,
                command_receive,
                "elgato-main",
            )
            await manager._reconcile_routing_current_state(reason="test snapshot")
            with anyio.fail_after(1):
                while device.clear_key.await_count < 1:
                    await anyio.sleep(0.01)
            tg.cancel_scope.cancel()

    assert "deck" not in manager._claims
    device.clear_key.assert_awaited_once()
    device.refresh.assert_awaited_once()


@pytest.mark.asyncio
async def test_direct_commands_apply_only_from_claiming_controller() -> None:
    class FakeDevice:
        id = "deck"

        def __init__(self) -> None:
            self.set_raster_frame = AsyncMock()
            self.clear_raster = AsyncMock()
            self.sleep_device = AsyncMock()
            self.wake_device = AsyncMock()
            self.clear_key = AsyncMock()
            self.refresh = AsyncMock()

    async with _deckr() as deckr:
        manager = _factory(deckr)
        manager._devices["deck"] = _device()
        manager._claims["deck"] = _claim()
        manager._controller_presence_sessions[controller_address("main")] = (
            "controller-session"
        )
        device = FakeDevice()
        command_send, command_receive = anyio.create_memory_object_stream(max_buffer_size=100)
        manager._command_streams["deck"] = command_send

        async with command_send, command_receive, anyio.create_task_group() as tg:
            tg.start_soon(
                _apply_device_commands,
                device,
                command_receive,
                "elgato-main",
            )
            await manager._route_command(
                _command_message("other", b"wrong")
            )
            await anyio.sleep(0.05)
            device.set_raster_frame.assert_not_awaited()

            await manager._route_command(
                _command_message("main", b"ok")
            )
            with anyio.fail_after(1):
                while device.set_raster_frame.await_count < 1:
                    await anyio.sleep(0.01)
            await manager._route_command(_power_command_message("main", "wake"))
            with anyio.fail_after(1):
                while device.wake_device.await_count < 1:
                    await anyio.sleep(0.01)
            tg.cancel_scope.cancel()

    device.set_raster_frame.assert_awaited_once_with("0,0", b"ok")
    device.wake_device.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_graceful_stop_revision_deletes_presence_and_inventory() -> None:
    async with _deckr() as deckr:
        manager = _factory(deckr)
        manager._devices["deck"] = _device()
        await manager._endpoint._ensure_presence()
        await manager._publish_inventory()

        assert (
            await deckr.state().get(
                presence_endpoint_key(
                    lane="hardware_messages",
                    endpoint=hardware_manager_address("elgato-main"),
                )
            )
        ) is not None
        assert (
            await deckr.state(DEFAULT_DISCOVERY_STATE_STORE_NAME).get(
                hardware_inventory_key("elgato-main")
            )
            is not None
        )

        await manager.stop()

        assert (
            await deckr.state().get(
                presence_endpoint_key(
                    lane="hardware_messages",
                    endpoint=hardware_manager_address("elgato-main"),
                )
            )
        ) is None
        assert (
            await deckr.state(DEFAULT_DISCOVERY_STATE_STORE_NAME).get(
                hardware_inventory_key("elgato-main")
            )
            is None
        )
