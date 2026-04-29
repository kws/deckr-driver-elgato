from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import anyio
import pytest
from deckr.contracts.lanes import CORE_LANE_CONTRACTS, LaneContractRegistry
from deckr.contracts.messages import (
    controller_address,
    hardware_manager_address,
)
from deckr.hardware import messages as hw_messages
from deckr.runtime import Deckr
from deckr.state import (
    DeviceClaim,
    EndpointPresence,
    HardwareInventory,
    StateUnavailable,
    hardware_inventory_key,
    presence_endpoint_key,
)
from memory_lane_substrate import MemoryLaneSubstrate

from deckr.drivers.elgato._discovery import _apply_device_commands, discover_loop
from deckr.drivers.elgato._factory import ElgatoDeviceFactory, driver_factory


class _FakeDeviceManager:
    def __init__(self, devices: list[object]) -> None:
        self._devices = devices

    def enumerate(self) -> list[object]:
        return self._devices


def _deckr() -> Deckr:
    lane_contracts = LaneContractRegistry(CORE_LANE_CONTRACTS.values())
    return Deckr(
        lane_contracts=lane_contracts,
        substrate=MemoryLaneSubstrate(lane_contracts=lane_contracts),
    )


def _device() -> hw_messages.HardwareDevice:
    return hw_messages.HardwareDevice(
        id="deck",
        name="Stream Deck",
        hid="hid:deck",
        fingerprint="fingerprint:deck",
        slots=[
            hw_messages.HardwareSlot(
                id="0,0",
                coordinates=hw_messages.HardwareCoordinates(column=0, row=0),
                image_format=hw_messages.HardwareImageFormat(width=72, height=72),
                gestures=("key_down", "key_up"),
            )
        ],
    )


def _factory(deckr: Deckr) -> ElgatoDeviceFactory:
    manager = ElgatoDeviceFactory(
        deckr.lane("hardware_messages"),
        deckr.state(),
        manager_id="elgato-main",
    )
    manager._endpoint = deckr.lane("hardware_messages").endpoint(
        hardware_manager_address("elgato-main")
    )
    return manager


def _claim(controller_id: str = "main", session_id: str = "controller-session"):
    return DeviceClaim(
        claimedByEndpoint=controller_address(controller_id),
        claimedBySessionId=session_id,
        timestamp=datetime.now(UTC),
        ttlSeconds=15,
    )


async def _put_controller_presence(
    deckr: Deckr,
    *,
    controller_id: str = "main",
    session_id: str = "controller-session",
) -> None:
    endpoint = controller_address(controller_id)
    await deckr.state().put(
        presence_endpoint_key(lane="hardware_messages", endpoint=endpoint),
        EndpointPresence(
            endpoint=endpoint,
            lane="hardware_messages",
            sessionId=session_id,
            timestamp=datetime.now(UTC),
            ttlSeconds=15,
            metadata={},
        ),
    )


def test_driver_factory_returns_elgato_device_factory() -> None:
    async def check() -> None:
        async with _deckr() as deckr:
            factory = driver_factory(
                deckr.lane("hardware_messages"),
                deckr.state(),
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
            hw_messages.hardware_input_message(
                manager_id="elgato-main",
                device_id="deck",
                body=hw_messages.DeviceConnectedMessage(device=_device()),
            )
        )
        entry = await deckr.state().get(hardware_inventory_key("elgato-main"))
        assert entry is not None
        inventory = HardwareInventory.model_validate(entry.value)
        assert set(inventory.devices) == {"deck"}
        assert inventory.devices["deck"].descriptor["id"] == "deck"

        await manager._handle_device_message(
            hw_messages.hardware_input_message(
                manager_id="elgato-main",
                device_id="deck",
                body=hw_messages.DeviceDisconnectedMessage(),
            )
        )
        entry = await deckr.state().get(hardware_inventory_key("elgato-main"))
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
            UnavailableState(),
            manager_id="elgato-main",
        )
        manager._endpoint = deckr.lane("hardware_messages").endpoint(
            hardware_manager_address("elgato-main")
        )

        await manager._handle_device_message(
            hw_messages.hardware_input_message(
                manager_id="elgato-main",
                device_id="deck",
                body=hw_messages.DeviceConnectedMessage(device=_device()),
            )
        )

    assert "deck" in manager._devices
    assert manager._inventory_revision is None


@pytest.mark.asyncio
async def test_presence_heartbeat_refreshes_aggregate_inventory() -> None:
    async with _deckr() as deckr:
        manager = _factory(deckr)
        manager._devices["deck"] = _device()

        async with anyio.create_task_group() as tg:
            tg.start_soon(manager._presence_loop)
            with anyio.fail_after(1):
                while True:
                    entry = await deckr.state().get(
                        hardware_inventory_key("elgato-main")
                    )
                    if entry is not None:
                        break
                    await anyio.sleep(0.01)
            tg.cancel_scope.cancel()

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
        main = deckr.lane("hardware_messages").endpoint(controller_address("main"))
        other = deckr.lane("hardware_messages").endpoint(controller_address("other"))

        async with main.subscribe() as main_stream, other.subscribe() as other_stream:
            await manager._handle_device_message(
                hw_messages.hardware_input_message(
                    manager_id="elgato-main",
                    device_id="deck",
                    body=hw_messages.KeyDownMessage(key_id="0,0"),
                )
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
        main = deckr.lane("hardware_messages").endpoint(controller_address("main"))

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
                hw_messages.hardware_input_message(
                    manager_id="elgato-main",
                    device_id="deck",
                    body=hw_messages.KeyDownMessage(key_id="0,0"),
                )
            )
            with anyio.move_on_after(0.05) as scope:
                await main_stream.receive()
            tg.cancel_scope.cancel()

    device.clear_key.assert_awaited_once()
    device.refresh.assert_awaited_once()
    assert scope.cancel_called


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

        main = deckr.lane("hardware_messages").endpoint(controller_address("main"))
        async with main.subscribe() as main_stream:
            await manager._handle_device_message(
                hw_messages.hardware_input_message(
                    manager_id="elgato-main",
                    device_id="deck",
                    body=hw_messages.KeyDownMessage(key_id="0,0"),
                )
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
                "ttlSeconds": 15,
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
            self.set_image = AsyncMock()
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
                hw_messages.hardware_command_for_control(
                    controller_id="other",
                    ref=hw_messages.HardwareControlRef(
                        manager_id="elgato-main",
                        device_id="deck",
                        control_id="0,0",
                        control_kind="slot",
                    ),
                    message_type=hw_messages.SET_IMAGE,
                    body=hw_messages.SetImageMessage(slot_id="0,0", image=b"wrong"),
                )
            )
            await anyio.sleep(0.05)
            device.set_image.assert_not_awaited()

            await manager._route_command(
                hw_messages.hardware_command_for_control(
                    controller_id="main",
                    ref=hw_messages.HardwareControlRef(
                        manager_id="elgato-main",
                        device_id="deck",
                        control_id="0,0",
                        control_kind="slot",
                    ),
                    message_type=hw_messages.SET_IMAGE,
                    body=hw_messages.SetImageMessage(slot_id="0,0", image=b"ok"),
                )
            )
            with anyio.fail_after(1):
                while device.set_image.await_count < 1:
                    await anyio.sleep(0.01)
            tg.cancel_scope.cancel()

    device.set_image.assert_awaited_once_with("0,0", b"ok")


@pytest.mark.asyncio
async def test_graceful_stop_revision_deletes_presence_and_inventory() -> None:
    async with _deckr() as deckr:
        manager = _factory(deckr)
        manager._devices["deck"] = _device()
        presence_entry = await deckr.state().put(
            presence_endpoint_key(
                lane="hardware_messages",
                endpoint=hardware_manager_address("elgato-main"),
            ),
            EndpointPresence(
                endpoint=hardware_manager_address("elgato-main"),
                lane="hardware_messages",
                sessionId=manager._session_id,
                timestamp=datetime.now(UTC),
                ttlSeconds=15,
                metadata={},
            ),
        )
        manager._presence_revision = presence_entry.revision
        await manager._publish_inventory()

        assert (
            await deckr.state().get(
                presence_endpoint_key(
                    lane="hardware_messages",
                    endpoint=hardware_manager_address("elgato-main"),
                )
            )
        ) is not None
        assert await deckr.state().get(hardware_inventory_key("elgato-main")) is not None

        await manager.stop()

        assert (
            await deckr.state().get(
                presence_endpoint_key(
                    lane="hardware_messages",
                    endpoint=hardware_manager_address("elgato-main"),
                )
            )
        ) is None
        assert await deckr.state().get(hardware_inventory_key("elgato-main")) is None
