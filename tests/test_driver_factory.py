from __future__ import annotations

from contextlib import asynccontextmanager

import anyio
import deckr.hardware.messages as hw_messages
import pytest
from deckr.beacon import (
    DEFAULT_BEACON_ADVERTISEMENT_STORE_NAME,
    BeaconDiscovery,
    BeaconService,
)
from deckr.components import RunContext
from deckr.concord import (
    DEFAULT_CONCORD_CONTRACT_STORE_NAME,
    DEFAULT_CONCORD_TOKEN_STORE_NAME,
    ConcordCoordinator,
    ConcordService,
    ContractValidityStatus,
)
from deckr.contracts.lanes import CORE_LANE_CONTRACTS, LaneContractRegistry
from deckr.contracts.messages import controller_address, hardware_manager_address
from deckr.hardware import (
    HARDWARE_CLAIM_PROFILE_ID,
    HARDWARE_FEATURE_ID,
    CapabilityDescriptor,
    ControlDescriptor,
    DeviceDescriptor,
    DeviceRef,
    HardwareBeaconPayload,
    HardwareClaimDevice,
    HardwareClaimTerms,
)
from deckr.runtime import Deckr
from memory_lane_substrate import MemoryLaneSubstrate

from deckr.drivers.elgato import _factory as factory_module
from deckr.drivers.elgato._discovery import ResetDeviceCommand

pytestmark = pytest.mark.asyncio


def _deckr() -> Deckr:
    registry = LaneContractRegistry(CORE_LANE_CONTRACTS.values())
    return Deckr(
        lane_contracts=registry,
        substrate=MemoryLaneSubstrate(lane_contracts=registry),
    )


def _beacon(deckr: Deckr) -> BeaconService:
    return BeaconService(
        BeaconDiscovery(deckr.state(DEFAULT_BEACON_ADVERTISEMENT_STORE_NAME))
    )


def _concord(deckr: Deckr) -> ConcordService:
    return ConcordService(
        ConcordCoordinator(
            deckr.state(DEFAULT_CONCORD_CONTRACT_STORE_NAME),
            deckr.state(DEFAULT_CONCORD_TOKEN_STORE_NAME),
        )
    )


def _device() -> DeviceDescriptor:
    return DeviceDescriptor(
        deviceId="deck",
        displayName="Test Deck",
        fingerprint="serial-a",
        controls=(
            ControlDescriptor(
                controlId="0,0",
                kind="key",
                outputCapabilities=(
                    CapabilityDescriptor(
                        capabilityId="raster.bitmap",
                        family="dev.deckr.output.raster",
                        type="bitmap",
                        direction="output",
                        access=("settable",),
                        commandTypes=("set_frame", "clear"),
                    ),
                ),
            ),
        ),
    )


class FakeDiscovery:
    def __init__(self) -> None:
        self.send, self.receive = anyio.create_memory_object_stream(20)
        self.kwargs = {}

    @asynccontextmanager
    async def __call__(self, **kwargs):
        self.kwargs = kwargs
        async with self.receive:
            yield self.receive


async def _claim(factory, concord: ConcordService, controller_endpoint):
    runtime = factory._runtime
    assert runtime is not None
    advertisement = runtime.advertisement
    assert advertisement is not None
    terms = HardwareClaimTerms(
        claimId="claim-1",
        controllerEndpoint=controller_endpoint.endpoint,
        managerEndpoint=hardware_manager_address("elgato-main"),
        managerAdvertisementId=advertisement.advertisement_id,
        devices=(
            HardwareClaimDevice(
                deviceRef=DeviceRef(
                    managerId="elgato-main",
                    deviceId="deck",
                    fingerprint="serial-a",
                ),
                instanceCount=1,
            ),
        ),
    )
    contract = await concord.create_contract(
        (controller_endpoint.endpoint, hardware_manager_address("elgato-main")),
        contract_id="claim-1",
        profile=HARDWARE_CLAIM_PROFILE_ID,
        terms=terms,
        created_by=controller_endpoint.endpoint,
    )
    await concord.attach(
        contract,
        controller_endpoint.endpoint,
        controller_endpoint.session_id,
    )
    await runtime.reconcile_claims(reason="test")
    return contract


async def test_elgato_advertises_hardware_and_routes_claimed_input(monkeypatch):
    fake_discovery = FakeDiscovery()
    monkeypatch.setattr(factory_module, "discover_elgato_devices", fake_discovery)
    deckr = _deckr()
    factory = factory_module.driver_factory(
        deckr.lane("hardware_messages"),
        _beacon(deckr),
        _concord(deckr),
        manager_id="elgato-main",
        labels={"room": "office"},
    )
    controller_cm = deckr.lane("hardware_messages").register_endpoint(
        controller_address("controller-main")
    )
    controller_endpoint = await controller_cm.__aenter__()
    try:
        async with anyio.create_task_group() as tg:
            await factory.start(RunContext(tg=tg, stopping=anyio.Event()))
            runtime = factory._runtime
            assert runtime is not None
            await fake_discovery.send.send(
                hw_messages.device_available_message(
                    manager_id="elgato-main",
                    sender_session_id=runtime.endpoint.session_id,
                    descriptor=_device(),
                )
            )
            with anyio.fail_after(1):
                while "deck" not in runtime.devices:
                    await anyio.sleep(0.01)

            candidates = await _beacon(deckr).find(HARDWARE_FEATURE_ID)
            assert len(candidates) == 1
            payload = HardwareBeaconPayload.model_validate(
                candidates[0].advertisement.payload
            )
            assert payload.labels == {"room": "office"}
            assert payload.devices["deck"].descriptor == _device()

            contract = await _claim(factory, _concord(deckr), controller_endpoint)
            assert (await _concord(deckr).validate(contract)).status == (
                ContractValidityStatus.VALID
            )

            async with controller_endpoint.subscribe() as stream:
                await fake_discovery.send.send(
                    hw_messages.control_input_message(
                        manager_id="elgato-main",
                        sender_session_id=runtime.endpoint.session_id,
                        device_id="deck",
                        fingerprint="serial-a",
                        control_id="0,0",
                        capability_id="raster.bitmap",
                        event_type="press",
                        value={"eventType": "press"},
                    )
                )
                with anyio.fail_after(1):
                    routed = await stream.receive()
            assert routed.recipient.endpoint == controller_endpoint.endpoint
            assert routed.recipient_session_id == controller_endpoint.session_id
            await factory.stop()
            tg.cancel_scope.cancel()
    finally:
        await controller_cm.__aexit__(None, None, None)


async def test_elgato_authorized_commands_and_claim_loss_reset(monkeypatch):
    fake_discovery = FakeDiscovery()
    monkeypatch.setattr(factory_module, "discover_elgato_devices", fake_discovery)
    deckr = _deckr()
    concord = _concord(deckr)
    factory = factory_module.driver_factory(
        deckr.lane("hardware_messages"),
        _beacon(deckr),
        concord,
        manager_id="elgato-main",
    )
    controller_cm = deckr.lane("hardware_messages").register_endpoint(
        controller_address("controller-main")
    )
    controller_endpoint = await controller_cm.__aenter__()
    command_send, command_receive = anyio.create_memory_object_stream(10)
    try:
        async with anyio.create_task_group() as tg:
            await factory.start(RunContext(tg=tg, stopping=anyio.Event()))
            runtime = factory._runtime
            assert runtime is not None
            await fake_discovery.send.send(
                hw_messages.device_available_message(
                    manager_id="elgato-main",
                    sender_session_id=runtime.endpoint.session_id,
                    descriptor=_device(),
                )
            )
            with anyio.fail_after(1):
                while "deck" not in runtime.devices:
                    await anyio.sleep(0.01)
            factory._command_streams["deck"] = command_send
            contract = await _claim(factory, concord, controller_endpoint)

            command = hw_messages.control_command_message(
                controller_id="controller-main",
                sender_session_id=controller_endpoint.session_id,
                manager_id="elgato-main",
                device_id="deck",
                control_id="0,0",
                capability_id="raster.bitmap",
                command_type="clear",
            )
            assert await runtime.handle_command(command)
            with anyio.fail_after(1):
                assert await command_receive.receive() == command

            await concord.cancel(contract, controller_endpoint.endpoint, reason="test")
            await runtime.reconcile_claims(reason="test cancel")
            with anyio.fail_after(1):
                assert isinstance(await command_receive.receive(), ResetDeviceCommand)

            await factory.stop()
            tg.cancel_scope.cancel()
    finally:
        await command_send.aclose()
        await command_receive.aclose()
        await controller_cm.__aexit__(None, None, None)
