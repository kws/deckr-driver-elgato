import logging
from contextlib import asynccontextmanager
from typing import Any

import anyio
from deckr.hardware import messages as hw_messages
from deckr.transports.bus import EventBus
from StreamDeck.DeviceManager import DeviceManager

from deckr.drivers.elgato._device import launch_device

logger = logging.getLogger(__name__)


@asynccontextmanager
async def discover_elgato_devices(event_bus: EventBus, *, manager_id: str):
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

    # Track if a device is currently connected (using a list to allow mutation from nested functions)
    device_connected = [False]

    async with anyio.create_task_group() as tg:
        tg.start_soon(discover_loop, discovery_send, device_connected)
        tg.start_soon(
            launcher_loop,
            discovery_receive,
            send_stream,
            event_bus,
            device_connected,
            manager_id,
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
    event_bus: EventBus,
    device_connected: list[bool],
    manager_id: str,
):
    """Launch devices as they are discovered."""
    async with anyio.create_task_group() as tg:
        async for device in receive_stream:
            tg.start_soon(
                device_loop,
                device,
                send_stream,
                event_bus,
                device_connected,
                manager_id,
            )


async def device_loop(
    device: Any,
    send_stream: anyio.abc.ObjectSendStream[Any],
    event_bus: EventBus,
    device_connected: list[bool],
    manager_id: str,
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

            logger.info("Device connected: %s", device_id)
            await send_stream.send(
                hw_messages.hardware_input_message(
                    manager_id=manager_id,
                    device_id=device_id,
                    body=hw_messages.DeviceConnectedMessage(
                        device=hw_messages.HardwareDevice(
                            id=my_device.id,
                            fingerprint=my_device.hid,
                            hid=my_device.hid,
                            slots=list(my_device.slots),
                            name=getattr(my_device, "name", None),
                        ),
                    ),
                )
            )
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
                    event_bus,
                    manager_id,
                )

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
                hw_messages.hardware_input_message(
                    manager_id=manager_id,
                    device_id=device_id,
                    body=hw_messages.DeviceDisconnectedMessage(),
                )
            )


async def _forward_device_events(
    device: Any,
    send_stream: anyio.abc.ObjectSendStream[Any],
    manager_id: str,
) -> None:
    async for event in device.subscribe():
        await send_stream.send(
            hw_messages.hardware_input_message(
                manager_id=manager_id,
                device_id=device.id,
                body=event,
            )
        )


async def _run_until_complete(cancel_scope, func, *args) -> None:
    try:
        await func(*args)
    finally:
        cancel_scope.cancel()


async def _apply_device_commands(
    device: Any,
    event_bus: EventBus,
    manager_id: str,
) -> None:
    async with event_bus.subscribe() as stream:
        async for envelope in stream:
            ref = hw_messages.hardware_device_ref_from_message(envelope)
            if ref != hw_messages.HardwareDeviceRef(
                manager_id=manager_id,
                device_id=device.id,
            ):
                continue
            message = hw_messages.hardware_body_from_message(envelope)
            if not isinstance(message, hw_messages.HARDWARE_COMMAND_MESSAGE_TYPES):
                continue
            if isinstance(message, hw_messages.SetImageMessage):
                await device.set_image(message.slot_id, message.image)
            elif isinstance(message, hw_messages.ClearSlotMessage):
                await device.clear_slot(message.slot_id)
            elif isinstance(message, hw_messages.SleepScreenMessage):
                await device.sleep_screen()
            elif isinstance(message, hw_messages.WakeScreenMessage):
                await device.wake_screen()
