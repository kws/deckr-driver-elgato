"""Elgato Stream Deck discovery and device supervision."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from typing import Any

import anyio
from deckr.contracts.messages import DeckrMessage
from deckr.hardware import messages as hw_messages
from deckr.hardware.runtime import HardwareManagerRuntime
from StreamDeck.DeviceManager import DeviceManager
from StreamDeck.Devices.StreamDeck import StreamDeck

from deckr.drivers.elgato._device import ElgatoDockDevice, launch_device

logger = logging.getLogger(__name__)

DeviceManagerFactory = Callable[[], DeviceManager]


class ElgatoDeviceSupervisor:
    """Poll, open, and supervise all attached Elgato Stream Deck devices."""

    def __init__(
        self,
        *,
        runtime: HardwareManagerRuntime,
        manager_id: str,
        device_manager_factory: DeviceManagerFactory = DeviceManager,
        poll_interval: float = 1.0,
    ) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be greater than zero")
        self._runtime = runtime
        self._manager_id = manager_id
        self._device_manager_factory = device_manager_factory
        self._poll_interval = poll_interval
        self._active_by_device_id: dict[str, ElgatoDockDevice] = {}
        self._active_by_token: dict[str, str] = {}
        self._active_fingerprints: set[str] = set()
        self._pending_tokens: set[str] = set()
        self._lock = anyio.Lock()
        self._stopping = anyio.Event()
        self._task_group: anyio.abc.TaskGroup | None = None

    @property
    def devices(self) -> dict[str, ElgatoDockDevice]:
        return dict(self._active_by_device_id)

    def start(self, task_group: anyio.abc.TaskGroup) -> None:
        self._task_group = task_group
        task_group.start_soon(self._poll_loop)

    async def stop(self) -> None:
        self._stopping.set()
        async with self._lock:
            devices = tuple(self._active_by_device_id.values())
        for device in devices:
            await device.mark_disconnected()

    async def handle_command(self, envelope: DeckrMessage) -> bool | None:
        ref = hw_messages.hardware_device_ref_from_message(envelope)
        if ref is None or ref.manager_id != self._manager_id:
            return None
        body = hw_messages.hardware_body_from_message(envelope)
        if not isinstance(body, hw_messages.ControlCommandMessage):
            return None
        async with self._lock:
            device = self._active_by_device_id.get(ref.device_id)
        if device is None:
            logger.debug(
                "Dropping command for closed Elgato device %s/%s",
                ref.manager_id,
                ref.device_id,
            )
            return False
        return await device.handle_command(envelope, manager_id=self._manager_id)

    async def reset_device(self, device_id: str) -> None:
        async with self._lock:
            device = self._active_by_device_id.get(device_id)
        if device is not None:
            await device.reset_outputs()

    async def _poll_loop(self) -> None:
        device_manager = self._device_manager_factory()
        while not self._stopping.is_set():
            try:
                devices = await anyio.to_thread.run_sync(
                    lambda: list(device_manager.enumerate())
                )
            except Exception:
                logger.warning("Could not enumerate Elgato devices", exc_info=True)
                await anyio.sleep(self._poll_interval)
                continue

            for device in devices:
                token = _raw_device_token(device)
                if await self._should_launch(token):
                    task_group = self._task_group
                    if task_group is not None:
                        task_group.start_soon(self._device_loop, device, token)
            await anyio.sleep(self._poll_interval)

    async def _should_launch(self, token: str) -> bool:
        async with self._lock:
            if token in self._pending_tokens or token in self._active_by_token:
                return False
            self._pending_tokens.add(token)
            return True

    async def _device_loop(self, raw_device: StreamDeck, token: str) -> None:
        device_id: str | None = None
        try:
            async with launch_device(raw_device) as device:
                if device is None:
                    return
                device_id = device.id
                if not await self._register_device(token, device):
                    return
                await self._runtime.set_device(device.descriptor)
                logger.info("Elgato device connected: %s", device.id)
                await self._run_open_device(device)
        except anyio.get_cancelled_exc_class():
            raise
        except Exception:
            logger.warning("Elgato device task failed", exc_info=True)
        finally:
            await self._unregister_device(token, device_id)

    async def _register_device(self, token: str, device: ElgatoDockDevice) -> bool:
        async with self._lock:
            self._pending_tokens.discard(token)
            if device.fingerprint in self._active_fingerprints:
                logger.debug("Ignoring duplicate Elgato device %s", device.fingerprint)
                return False
            self._active_by_token[token] = device.id
            self._active_by_device_id[device.id] = device
            self._active_fingerprints.add(device.fingerprint)
            return True

    async def _unregister_device(self, token: str, device_id: str | None) -> None:
        removed = False
        async with self._lock:
            self._pending_tokens.discard(token)
            mapped_device_id = self._active_by_token.pop(token, None)
            device_id = device_id or mapped_device_id
            device = (
                self._active_by_device_id.pop(device_id, None)
                if device_id is not None
                else None
            )
            if device is not None:
                self._active_fingerprints.discard(device.fingerprint)
                removed = True
        if removed and device_id is not None:
            await self._runtime.remove_device(device_id, reason="disconnected")
            logger.info("Elgato device disconnected: %s", device_id)

    async def _run_open_device(self, device: ElgatoDockDevice) -> None:
        async with anyio.create_task_group() as tg:
            tg.start_soon(
                _run_until_complete,
                tg.cancel_scope,
                self._forward_input,
                device,
            )
            tg.start_soon(
                _run_until_complete,
                tg.cancel_scope,
                self._monitor_connection,
                device,
            )

    async def _forward_input(self, device: ElgatoDockDevice) -> None:
        async for event in device.subscribe():
            await self._runtime.handle_hardware_message(
                hw_messages.control_input_message(
                    manager_id=self._manager_id,
                    sender_session_id=self._runtime.endpoint.session_id,
                    device_id=device.id,
                    fingerprint=device.fingerprint,
                    control_id=event.control_id,
                    capability_id=event.capability_id,
                    event_type=event.event_type,
                    value=event.value,
                    sources=event.sources,
                )
            )

    async def _monitor_connection(self, device: ElgatoDockDevice) -> None:
        while not self._stopping.is_set() and device.is_connected():
            await anyio.sleep(self._poll_interval)
        await device.mark_disconnected()


async def _run_until_complete(cancel_scope: anyio.CancelScope, func, *args) -> None:
    try:
        await func(*args)
    finally:
        cancel_scope.cancel()


def _raw_device_token(device: Any) -> str:
    transport = getattr(device, "device", None)
    path = getattr(transport, "path", None)
    if path is not None:
        try:
            value = path()
        except Exception:
            value = None
        if value is not None:
            return f"path:{value}"
    vendor = _safe_call(device, "vendor_id")
    product = _safe_call(device, "product_id")
    if vendor is not None and product is not None:
        return f"usb:{vendor}:{product}:{id(device)}"
    return "object:" + hashlib.sha256(str(id(device)).encode()).hexdigest()[:16]


def _safe_call(device: Any, method_name: str) -> Any:
    method = getattr(device, method_name, None)
    if method is None:
        return None
    try:
        return method()
    except Exception:
        return None
