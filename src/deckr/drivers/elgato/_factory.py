from __future__ import annotations

import logging
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime

import anyio
from deckr.components import (
    BaseComponent,
    ComponentContext,
    ComponentDefinition,
    ComponentManifest,
    RunContext,
)
from deckr.contracts.messages import (
    DeckrMessage,
    EndpointAddress,
    EndpointTarget,
    endpoint_target,
    hardware_manager_address,
)
from deckr.hardware import messages as hw_messages
from deckr.hardware.descriptors import DeviceDescriptor, DeviceRef
from deckr.lanes import Lane, RegisteredEndpointLane
from deckr.state import (
    DeviceClaim,
    EndpointPresence,
    HardwareInventory,
    HardwareInventoryDevice,
    StateConflict,
    StateStore,
    StateUnavailable,
    encode_key_token,
    hardware_inventory_key,
    parse_device_claim_key,
    parse_presence_endpoint_key,
)

from deckr.drivers.elgato._discovery import (
    DeviceCommand,
    ResetDeviceCommand,
    discover_elgato_devices,
)

logger = logging.getLogger(__name__)

INVENTORY_HEARTBEAT_SECONDS = 5.0
INVENTORY_TTL_SECONDS = 15
_STATE_RECONCILE_SECONDS = 1.0
_WATCH_RETRY_SECONDS = 1.0
_CONTROLLER_PRESENCE_PREFIX = ".".join(
    (
        "presence",
        "endpoint",
        encode_key_token("hardware_messages"),
        encode_key_token("controller"),
        "",
    )
)


class ElgatoDeviceFactory(BaseComponent):
    def __init__(self, hardware_lane: Lane, state: StateStore, *, manager_id: str):
        super().__init__("elgato_device_factory")
        self._hardware_lane = hardware_lane
        self._state = state
        self.manager_id = manager_id
        self._session_id = ""
        self._cancel_scope: anyio.CancelScope | None = None
        self._endpoint_cm: (
            AbstractAsyncContextManager[RegisteredEndpointLane] | None
        ) = None
        self._endpoint: RegisteredEndpointLane | None = None
        self._devices: dict[str, DeviceDescriptor] = {}
        self._claims: dict[str, DeviceClaim] = {}
        self._controller_presence_sessions: dict[EndpointAddress, str] = {}
        self._unroutable_devices: set[str] = set()
        self._command_streams: dict[str, anyio.abc.ObjectSendStream[DeviceCommand]] = {}
        self._inventory_revision: int | None = None
        self._routing_reconcile_lock = anyio.Lock()

    async def start(self, ctx: RunContext) -> None:
        try:
            self._endpoint_cm = self._hardware_lane.register_endpoint(
                hardware_manager_address(self.manager_id),
                metadata={"runtime": "deckr-driver-elgato-python"},
                task_group=ctx.tg,
            )
            self._endpoint = await self._endpoint_cm.__aenter__()
            self._session_id = self._endpoint.session_id
            self._cancel_scope = ctx.tg.cancel_scope
            await self._publish_inventory_safely()
            ctx.tg.start_soon(self._inventory_refresh_loop)
            ctx.tg.start_soon(self._command_subscription_loop)
            ctx.tg.start_soon(self._claim_watch_loop)
            ctx.tg.start_soon(self._controller_presence_loop)
            ctx.tg.start_soon(self._routing_reconciliation_loop)
            ctx.tg.start_soon(self._discovery_loop)
        except BaseException:
            with anyio.CancelScope(shield=True):
                await self._withdraw_inventory()
                await self._close_endpoint()
            raise

    async def stop(self) -> None:
        with anyio.CancelScope(shield=True):
            if self._cancel_scope is not None:
                self._cancel_scope.cancel()
            self._devices.clear()
            self._claims.clear()
            self._unroutable_devices.clear()
            await self._withdraw_inventory()
            await self._close_endpoint()

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
        if self._endpoint is None:
            return
        event = hw_messages.hardware_body_from_message(message)
        ref = hw_messages.hardware_device_ref_from_message(message)
        if ref is None:
            return
        if isinstance(event, hw_messages.DeviceAvailableMessage):
            self._devices[ref.device_id] = event.descriptor
            await self._publish_inventory_safely()
            await self._endpoint.publish(message)
            return
        if isinstance(event, hw_messages.DeviceDescriptorChangedMessage):
            self._devices[ref.device_id] = event.descriptor
            await self._publish_inventory_safely()
            await self._endpoint.publish(message)
            return
        if isinstance(event, hw_messages.DeviceUnavailableMessage):
            self._devices.pop(ref.device_id, None)
            self._claims.pop(ref.device_id, None)
            self._unroutable_devices.discard(ref.device_id)
            await self._publish_inventory_safely()
            await self._endpoint.publish(message)
            return
        if not isinstance(
            event,
            hw_messages.ControlInputMessage | hw_messages.CapabilityStateChangedMessage,
        ):
            return
        if ref.device_id not in self._devices:
            logger.debug("Dropping input for unknown Elgato device %s", ref.device_id)
            return
        recipient = self._claim_recipient(ref.device_id)
        if recipient is None:
            logger.debug(
                "Dropping unclaimed Elgato input for %s/%s",
                ref.manager_id,
                ref.device_id,
            )
            return
        await self._endpoint.publish(
            hw_messages.hardware_message(
                sender=self._endpoint.endpoint,
                sender_session_id=self._endpoint.session_id,
                recipient=endpoint_target(recipient),
                message_type=message.message_type,
                body=event,
                subject=message.subject,
                causation_id=message.causation_id,
            )
        )

    async def _publish_inventory(self) -> None:
        if self._endpoint is None:
            return
        entry = await self._state.put(
            hardware_inventory_key(self.manager_id),
            HardwareInventory(
                managerId=self.manager_id,
                managerEndpoint=self._endpoint.endpoint,
                sessionId=self._session_id,
                timestamp=datetime.now(UTC),
                ttlSeconds=INVENTORY_TTL_SECONDS,
                devices={
                    device_id: HardwareInventoryDevice(
                        deviceRef=DeviceRef(
                            managerId=self.manager_id,
                            deviceId=device_id,
                            fingerprint=device.fingerprint,
                        ),
                        descriptor=device,
                    )
                    for device_id, device in sorted(self._devices.items())
                },
            ),
            ttl=INVENTORY_TTL_SECONDS,
        )
        self._inventory_revision = entry.revision

    async def _publish_inventory_safely(self) -> None:
        try:
            await self._publish_inventory()
        except StateUnavailable:
            logger.warning(
                "Elgato inventory current state is unavailable; heartbeat will retry",
                exc_info=True,
            )

    async def _inventory_refresh_loop(self) -> None:
        while True:
            await anyio.sleep(INVENTORY_HEARTBEAT_SECONDS)
            await self._publish_inventory_safely()

    async def _withdraw_inventory(self) -> None:
        revision = self._inventory_revision
        if revision is None:
            return
        with anyio.CancelScope(shield=True):
            try:
                await self._state.delete(
                    hardware_inventory_key(self.manager_id),
                    revision=revision,
                )
                self._inventory_revision = None
            except StateConflict:
                logger.debug("Elgato inventory changed before withdrawal")
            except StateUnavailable:
                logger.warning("Failed to withdraw Elgato inventory", exc_info=True)

    async def _claim_watch_loop(self) -> None:
        prefix = f"claim.device.{encode_key_token(self.manager_id)}."
        while True:
            try:
                async with self._state.watch(prefix) as stream:
                    async for change in stream:
                        parsed = parse_device_claim_key(change.key)
                        if parsed is None:
                            continue
                        manager_id, _device_id = parsed
                        if manager_id != self.manager_id:
                            continue
                        await self._reconcile_routing_current_state(
                            reason="device claim watch"
                        )
            except StateUnavailable:
                logger.warning(
                    "Elgato device claim state is unavailable; watch will retry",
                    exc_info=True,
                )
                await anyio.sleep(_WATCH_RETRY_SECONDS)

    async def _controller_presence_loop(self) -> None:
        while True:
            try:
                async with self._state.watch(_CONTROLLER_PRESENCE_PREFIX) as stream:
                    async for change in stream:
                        parsed = parse_presence_endpoint_key(change.key)
                        if parsed is None:
                            continue
                        lane, endpoint = parsed
                        if lane != "hardware_messages" or endpoint.family != "controller":
                            continue
                        await self._reconcile_routing_current_state(
                            reason="controller presence watch"
                        )
            except StateUnavailable:
                logger.warning(
                    "Controller endpoint presence state is unavailable; watch will retry",
                    exc_info=True,
                )
                await anyio.sleep(_WATCH_RETRY_SECONDS)

    async def _routing_reconciliation_loop(self) -> None:
        while True:
            try:
                await self._reconcile_routing_current_state(reason="broker snapshot")
            except StateUnavailable:
                logger.warning(
                    "Elgato routing current state unavailable; reconciliation will retry",
                    exc_info=True,
                )
            await anyio.sleep(_STATE_RECONCILE_SECONDS)

    async def _reconcile_routing_current_state(self, *, reason: str) -> None:
        async with self._routing_reconcile_lock:
            await self._reconcile_routing_current_state_locked(reason=reason)

    async def _reconcile_routing_current_state_locked(self, *, reason: str) -> None:
        claim_prefix = f"claim.device.{encode_key_token(self.manager_id)}."
        claim_entries = await self._state.items(claim_prefix)
        presence_entries = await self._state.items(_CONTROLLER_PRESENCE_PREFIX)

        next_claims: dict[str, DeviceClaim] = {}
        invalid_claim_devices: set[str] = set()
        next_controller_sessions: dict[EndpointAddress, str] = {}

        for entry in claim_entries:
            parsed = parse_device_claim_key(entry.key)
            if parsed is None:
                continue
            manager_id, device_id = parsed
            if manager_id != self.manager_id:
                continue
            try:
                next_claims[device_id] = DeviceClaim.model_validate(entry.value)
            except ValueError:
                logger.warning("Ignoring invalid Elgato device claim %s", entry.key)
                invalid_claim_devices.add(device_id)

        for entry in presence_entries:
            parsed = parse_presence_endpoint_key(entry.key)
            if parsed is None:
                continue
            lane, endpoint = parsed
            if lane != "hardware_messages" or endpoint.family != "controller":
                continue
            try:
                presence = EndpointPresence.model_validate(entry.value)
            except ValueError:
                logger.warning("Ignoring invalid controller presence %s", entry.key)
                continue
            if presence.endpoint != endpoint or presence.lane != lane:
                logger.warning(
                    "Ignoring controller presence %s with mismatched payload",
                    entry.key,
                )
                continue
            next_controller_sessions[endpoint] = presence.session_id

        logger.debug("Reconciling Elgato routing current state via %s", reason)
        devices_to_reset = self._devices_to_reset_for_routing_snapshot(
            next_claims,
            next_controller_sessions,
            invalid_claim_devices,
        )
        self._claims = next_claims
        self._controller_presence_sessions = next_controller_sessions
        self._unroutable_devices = {
            device_id
            for device_id, claim in next_claims.items()
            if _claim_recipient(claim, next_controller_sessions) is None
        }
        for device_id in sorted(devices_to_reset):
            await self._reset_device(device_id)

    def _devices_to_reset_for_routing_snapshot(
        self,
        next_claims: dict[str, DeviceClaim],
        next_controller_sessions: dict[EndpointAddress, str],
        invalid_claim_devices: set[str],
    ) -> set[str]:
        devices_to_reset = set(invalid_claim_devices)
        for device_id, old_claim in self._claims.items():
            next_claim = next_claims.get(device_id)
            if next_claim is None:
                devices_to_reset.add(device_id)
                continue
            if _claim_route_identity(old_claim) != _claim_route_identity(next_claim):
                devices_to_reset.add(device_id)
                continue
            if (
                _claim_recipient(old_claim, self._controller_presence_sessions)
                is not None
                and _claim_recipient(next_claim, next_controller_sessions) is None
            ):
                devices_to_reset.add(device_id)

        for device_id, next_claim in next_claims.items():
            if (
                device_id not in self._claims
                and _claim_recipient(next_claim, next_controller_sessions) is None
            ):
                devices_to_reset.add(device_id)
        return devices_to_reset

    def _claim_recipient(self, device_id: str) -> EndpointAddress | None:
        claim = self._claims.get(device_id)
        if claim is None:
            return None
        return _claim_recipient(claim, self._controller_presence_sessions)

    async def _reset_device(self, device_id: str) -> None:
        stream = self._command_streams.get(device_id)
        if stream is None:
            return
        try:
            await stream.send(ResetDeviceCommand())
        except (anyio.BrokenResourceError, anyio.ClosedResourceError):
            logger.debug("Could not reset closed Elgato device session %s", device_id)

    async def _command_subscription_loop(self) -> None:
        if self._endpoint is None:
            return
        async with self._endpoint.subscribe() as stream:
            async for envelope in stream:
                await self._route_command(envelope)

    async def _route_command(self, envelope: DeckrMessage) -> None:
        if self._endpoint is None:
            return
        if (
            not isinstance(envelope.recipient, EndpointTarget)
            or envelope.recipient.endpoint != self._endpoint.endpoint
        ):
            return
        ref = hw_messages.hardware_device_ref_from_message(envelope)
        if ref is None or ref.manager_id != self.manager_id:
            return
        message = hw_messages.hardware_body_from_message(envelope)
        if not isinstance(
            message,
            hw_messages.ControlCommandMessage | hw_messages.CapabilityStateRequestMessage,
        ):
            return
        if ref.device_id not in self._devices:
            logger.debug(
                "Dropping command for unknown Elgato device %s/%s",
                ref.manager_id,
                ref.device_id,
            )
            return
        if self._claim_recipient(ref.device_id) != envelope.sender:
            logger.debug(
                "Dropping unroutable Elgato command for %s/%s from %s",
                ref.manager_id,
                ref.device_id,
                envelope.sender,
            )
            return
        command_stream = self._command_streams.get(ref.device_id)
        if command_stream is None:
            logger.debug(
                "Dropping command for closed Elgato device %s/%s",
                ref.manager_id,
                ref.device_id,
            )
            return
        await command_stream.send(envelope)


def _claim_route_identity(claim: DeviceClaim) -> tuple[EndpointAddress, str]:
    return claim.claimed_by_endpoint, claim.claimed_by_session_id


def _claim_recipient(
    claim: DeviceClaim,
    controller_presence_sessions: dict[EndpointAddress, str],
) -> EndpointAddress | None:
    session_id = controller_presence_sessions.get(claim.claimed_by_endpoint)
    if session_id != claim.claimed_by_session_id:
        return None
    return claim.claimed_by_endpoint


def driver_factory(
    hardware_lane: Lane,
    state: StateStore,
    *,
    manager_id: str,
) -> ElgatoDeviceFactory:
    return ElgatoDeviceFactory(
        hardware_lane=hardware_lane,
        state=state,
        manager_id=manager_id,
    )


def component_factory(context: ComponentContext) -> ElgatoDeviceFactory:
    source = dict(context.raw_config)
    manager_id = str(source.get("manager_id", "")).strip()
    if not manager_id:
        raise ValueError("deckr.drivers.elgato requires manager_id")
    return driver_factory(
        context.require_lane("hardware_messages"),
        context.state(),
        manager_id=manager_id,
    )


component = ComponentDefinition(
    manifest=ComponentManifest(
        component_id="deckr.drivers.elgato",
        config_prefix="deckr.drivers.elgato",
        consumes=("hardware_messages",),
        publishes=("hardware_messages",),
    ),
    factory=component_factory,
)
