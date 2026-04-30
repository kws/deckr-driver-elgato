import base64
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import anyio
from deckr.contracts.messages import DeckrMessage
from deckr.hardware import messages as hw_messages
from StreamDeck.DeviceManager import DeviceManager

from deckr.drivers.elgato._device import launch_device

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ResetDeviceCommand:
    pass


DeviceCommand = DeckrMessage | ResetDeviceCommand


@asynccontextmanager
async def discover_elgato_devices(
    *,
    manager_id: str,
    command_streams: dict[str, anyio.abc.ObjectSendStream[DeviceCommand]] | None = None,
):
    """
    The discovery loop manages StreamDeck device connections. It discovers the first
    available device and opens it. If the device disconnects, it will be re-discovered
    and re-opened.

    Only one device can be connected at a time. When a device is successfully opened,
    a connection event is sent to the event stream. When the device disconnects, a
    disconnected event is sent.
    """
    send_stream, receive_stream = anyio.create_memory_object_stream[Any](
        max_buffer_size=100
    )
    discovery_send, discovery_receive = anyio.create_memory_object_stream[Any](
        max_buffer_size=100
    )
    if command_streams is None:
        command_streams = {}

    # Track if a device is currently connected (using a list to allow mutation from nested functions)
    device_connected = [False]

    async with anyio.create_task_group() as tg:
        tg.start_soon(discover_loop, discovery_send, device_connected)
        tg.start_soon(
            launcher_loop,
            discovery_receive,
            send_stream,
            device_connected,
            manager_id,
            command_streams,
        )
        yield receive_stream


async def discover_loop(
    send_stream: anyio.abc.ObjectSendStream[Any],
    device_connected: list[bool],
):
    """Poll for StreamDeck devices and send the first one if no device is connected."""
    device_manager = DeviceManager()

    while True:
        try:
            devices = list(device_manager.enumerate())

            # Only send a device if we don't have one connected
            if not device_connected[0] and len(devices) > 0:
                await send_stream.send(devices[0])
        except Exception as e:
            logger.error(f"Error in discovery loop: {e}", exc_info=True)

        await anyio.sleep(1)


async def launcher_loop(
    receive_stream: anyio.abc.ObjectReceiveStream[Any],
    send_stream: anyio.abc.ObjectSendStream[Any],
    device_connected: list[bool],
    manager_id: str,
    command_streams: dict[str, anyio.abc.ObjectSendStream[DeviceCommand]],
):
    """Launch devices as they are discovered."""
    async with anyio.create_task_group() as tg:
        async for device in receive_stream:
            tg.start_soon(
                device_loop,
                device,
                send_stream,
                device_connected,
                manager_id,
                command_streams,
            )


async def device_loop(
    device: Any,
    send_stream: anyio.abc.ObjectSendStream[Any],
    device_connected: list[bool],
    manager_id: str,
    command_streams: dict[str, anyio.abc.ObjectSendStream[DeviceCommand]],
):
    """Handle a single device's lifecycle."""
    cancelled = anyio.get_cancelled_exc_class()
    device_id = None

    try:
        async with launch_device(device) as my_device:
            if my_device is None:
                return

            device_id = my_device.id
            device_connected[0] = True  # Signal device is connected
            command_send, command_receive = anyio.create_memory_object_stream[
                DeviceCommand
            ](max_buffer_size=100)
            command_streams[device_id] = command_send

            logger.info("Device connected: %s", device_id)
            await send_stream.send(
                hw_messages.device_available_message(
                    manager_id=manager_id,
                    descriptor=my_device.descriptor,
                )
            )
            async with command_send, command_receive:
                try:
                    async with anyio.create_task_group() as tg:
                        tg.start_soon(
                            _run_until_complete,
                            tg.cancel_scope,
                            _forward_device_events,
                            my_device,
                            send_stream,
                            manager_id,
                        )
                        tg.start_soon(
                            _run_until_complete,
                            tg.cancel_scope,
                            _apply_device_commands,
                            my_device,
                            command_receive,
                            manager_id,
                        )
                finally:
                    command_streams.pop(device_id, None)

    except cancelled as e:
        raise e
    except Exception as e:
        logger.info("Device error: %s", e)
        if logger.isEnabledFor(logging.DEBUG):
            logger.exception("Device error: %s", e, exc_info=True)
    finally:
        # Signal device disconnected
        device_connected[0] = False
        if device_id is not None:
            await send_stream.send(
                hw_messages.device_unavailable_message(
                    manager_id=manager_id,
                    device_id=device_id,
                    reason="disconnected",
                )
            )


async def _forward_device_events(
    device: Any,
    send_stream: anyio.abc.ObjectSendStream[Any],
    manager_id: str,
) -> None:
    async for event in device.subscribe():
        await send_stream.send(
            hw_messages.control_input_message(
                manager_id=manager_id,
                device_id=device.id,
                fingerprint=device.hid,
                control_id=event.control_id,
                capability_id=event.capability_id,
                event_type=event.event_type,
                value=event.value,
            )
        )


async def _run_until_complete(cancel_scope, func, *args) -> None:
    try:
        await func(*args)
    finally:
        cancel_scope.cancel()


async def _apply_device_commands(
    device: Any,
    command_stream: anyio.abc.ObjectReceiveStream[DeviceCommand],
    manager_id: str,
) -> None:
    async for command in command_stream:
        if isinstance(command, ResetDeviceCommand):
            await device.clear_key()
            await device.refresh()
            continue
        envelope = command
        ref = hw_messages.hardware_device_ref_from_message(envelope)
        if ref is None or ref.manager_id != manager_id or ref.device_id != device.id:
            continue
        message = hw_messages.hardware_body_from_message(envelope)
        if not isinstance(message, hw_messages.ControlCommandMessage):
            continue
        if message.capability_id == "device.power":
            if message.command_type == "wake":
                await device.wake_device()
            elif message.command_type == "sleep":
                await device.sleep_device()
            continue
        if message.capability_id != "raster.bitmap" or message.control_id is None:
            continue
        if message.command_type == "set_frame":
            encoded = message.params.get("image")
            if not isinstance(encoded, str):
                continue
            await device.set_raster_frame(
                message.control_id,
                base64.b64decode(encoded),
            )
        elif message.command_type == "clear":
            await device.clear_raster(message.control_id)
