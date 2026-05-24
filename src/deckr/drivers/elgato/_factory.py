from __future__ import annotations

import logging
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager

import anyio
import deckr.hardware.messages as hw_messages
from deckr.beacon import (
    BEACON_ADVERTISEMENT_STORE_POLICY,
    DEFAULT_BEACON_ADVERTISEMENT_STORE_NAME,
    BeaconDiscovery,
    BeaconService,
)
from deckr.components import (
    BaseComponent,
    ComponentContext,
    ComponentDefinition,
    ComponentManifest,
    RunContext,
)
from deckr.concord import (
    CONCORD_CONTRACT_STORE_POLICY,
    CONCORD_TOKEN_STORE_POLICY,
    DEFAULT_CONCORD_CONTRACT_STORE_NAME,
    DEFAULT_CONCORD_TOKEN_STORE_NAME,
    ConcordCoordinator,
    ConcordService,
)
from deckr.contracts.messages import DeckrMessage, hardware_manager_address
from deckr.hardware.runtime import HardwareManagerRuntime
from deckr.lanes import Lane, RegisteredEndpointLane

from deckr.drivers.elgato._discovery import (
    DeviceCommand,
    ResetDeviceCommand,
    discover_elgato_devices,
)

logger = logging.getLogger(__name__)


def _labels_from_config(config: Mapping[str, object] | None) -> dict[str, str]:
    raw = dict(config or {}).get("labels", {})
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("Elgato manager config.labels must be a table")
    labels: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Elgato manager config.labels keys must be strings")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Elgato manager config.labels.{key} must be a non-empty string"
            )
        labels[key.strip()] = value.strip()
    return labels


class ElgatoDeviceFactory(BaseComponent):
    def __init__(
        self,
        hardware_lane: Lane,
        beacon: BeaconService,
        concord: ConcordService,
        *,
        manager_id: str,
        labels: Mapping[str, str] | None = None,
    ):
        super().__init__("elgato_device_factory")
        self._hardware_lane = hardware_lane
        self._beacon = beacon
        self._concord = concord
        self.manager_id = manager_id
        self._labels = dict(labels or {})
        self._cancel_scope: anyio.CancelScope | None = None
        self._endpoint_cm: (
            AbstractAsyncContextManager[RegisteredEndpointLane] | None
        ) = None
        self._endpoint: RegisteredEndpointLane | None = None
        self._runtime: HardwareManagerRuntime | None = None
        self._command_streams: dict[str, anyio.abc.ObjectSendStream[DeviceCommand]] = {}

    async def start(self, ctx: RunContext) -> None:
        try:
            self._endpoint_cm = self._hardware_lane.register_endpoint(
                hardware_manager_address(self.manager_id),
                metadata={"runtime": "deckr-driver-elgato-python"},
                task_group=ctx.tg,
            )
            self._endpoint = await self._endpoint_cm.__aenter__()
            self._cancel_scope = ctx.tg.cancel_scope
            self._runtime = HardwareManagerRuntime(
                endpoint=self._endpoint,
                beacon=self._beacon,
                concord=self._concord,
                manager_id=self.manager_id,
                labels=self._labels,
                command_handler=self._send_device_command,
                reset_handler=self._reset_device,
            )
            await self._runtime.start(ctx.tg)
            ctx.tg.start_soon(self._discovery_loop)
        except BaseException:
            with anyio.CancelScope(shield=True):
                await self._stop_runtime()
                await self._close_endpoint()
            raise

    async def stop(self) -> None:
        with anyio.CancelScope(shield=True):
            if self._cancel_scope is not None:
                self._cancel_scope.cancel()
            self._command_streams.clear()
            await self._stop_runtime()
            await self._close_endpoint()

    async def _stop_runtime(self) -> None:
        runtime = self._runtime
        self._runtime = None
        if runtime is not None:
            await runtime.stop()

    async def _close_endpoint(self) -> None:
        endpoint_cm = self._endpoint_cm
        self._endpoint_cm = None
        self._endpoint = None
        if endpoint_cm is not None:
            await endpoint_cm.__aexit__(None, None, None)

    async def _discovery_loop(self) -> None:
        if self._endpoint is None:
            return
        async with discover_elgato_devices(
            manager_id=self.manager_id,
            sender_session_id=self._endpoint.session_id,
            command_streams=self._command_streams,
        ) as stream:
            async for message in stream:
                await self._handle_device_message(message)

    async def _handle_device_message(self, message: DeckrMessage) -> None:
        if self._runtime is None:
            return
        await self._runtime.handle_hardware_message(message)

    async def _send_device_command(self, envelope: DeckrMessage) -> bool | None:
        ref = hw_messages.hardware_device_ref_from_message(envelope)
        if ref is None or ref.manager_id != self.manager_id:
            return None
        message = hw_messages.hardware_body_from_message(envelope)
        if not isinstance(message, hw_messages.ControlCommandMessage):
            return None
        command_stream = self._command_streams.get(ref.device_id)
        if command_stream is None:
            logger.debug(
                "Dropping command for closed Elgato device %s/%s",
                ref.manager_id,
                ref.device_id,
            )
            return False
        await command_stream.send(envelope)
        return True

    async def _reset_device(self, device_id: str) -> None:
        stream = self._command_streams.get(device_id)
        if stream is None:
            return
        await stream.send(ResetDeviceCommand())


def driver_factory(
    hardware_lane: Lane,
    beacon: BeaconService,
    concord: ConcordService,
    *,
    manager_id: str,
    labels: Mapping[str, str] | None = None,
) -> ElgatoDeviceFactory:
    return ElgatoDeviceFactory(
        hardware_lane=hardware_lane,
        beacon=beacon,
        concord=concord,
        manager_id=manager_id,
        labels=labels,
    )


def component_factory(context: ComponentContext) -> ElgatoDeviceFactory:
    return driver_factory(
        context.require_lane("hardware_messages"),
        BeaconService(
            BeaconDiscovery(
                context.state(
                    DEFAULT_BEACON_ADVERTISEMENT_STORE_NAME,
                    policy=BEACON_ADVERTISEMENT_STORE_POLICY,
                )
            )
        ),
        ConcordService(
            ConcordCoordinator(
                context.state(
                    DEFAULT_CONCORD_CONTRACT_STORE_NAME,
                    policy=CONCORD_CONTRACT_STORE_POLICY,
                ),
                context.state(
                    DEFAULT_CONCORD_TOKEN_STORE_NAME,
                    policy=CONCORD_TOKEN_STORE_POLICY,
                ),
            )
        ),
        manager_id=context.require_endpoint_id("hardware_manager"),
        labels=_labels_from_config(context.config),
    )


component = ComponentDefinition(
    manifest=ComponentManifest(
        component_id="dev.deckr.hardware.elgato",
        consumes=("hardware_messages",),
        publishes=("hardware_messages",),
        endpoint_slots=("hardware_manager",),
        role="hardware_manager",
    ),
    factory=component_factory,
)
