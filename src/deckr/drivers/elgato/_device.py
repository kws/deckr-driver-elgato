"""Elgato Stream Deck device implementation."""

from contextlib import asynccontextmanager
from typing import AsyncIterator
import logging

import anyio

import deckr.hardware.events as hw_events

from StreamDeck.Devices.StreamDeck import StreamDeck

logger = logging.getLogger(__name__)


class ElgatoDockDevice:
    """Elgato Stream Deck device implementation.
    This uses the third-party library python-elgato-streamdeck to communicate with the device.
    """

    def __init__(self, device: StreamDeck):
        """Initialize the Elgato dock device.

        Args:
            device: StreamDeck device object
        """
        self._device = device
        self._rows, self._cols = device.key_layout()
        self._slots = self._create_slots()
        self._slot_to_key_map = self._create_slot_to_key_map()
        self._key_to_slot_map = {v: k for k, v in self._slot_to_key_map.items()}
        self._event_send, self._event_receive = anyio.create_memory_object_stream[
            hw_events.HardwareEvent
        ](max_buffer_size=100)
        self._device_id = None
        self._hid = None
        self._disconnected = False

    def _create_slots(self) -> list[hw_events.HWSlot]:
        """Create HWSlot objects for all keys on the device."""
        slots = []
        for row in range(self._rows):
            for col in range(self._cols):
                slot_id = f"{col},{row}"
                slots.append(
                    hw_events.HWSlot(
                        id=slot_id,
                        coordinates=hw_events.Coordinates(column=col, row=row),
                        image_format=hw_events.HWSImageFormat(
                            width=72,
                            height=72,
                            format="JPEG",
                            rotation=180,
                            format_options={"quality": 10},
                        ),
                    )
                )
        return slots

    def _create_slot_to_key_map(self) -> dict[str, int]:
        """Create mapping from slot_id (e.g., "0,0") to key index."""
        mapping = {}
        for row in range(self._rows):
            for col in range(self._cols):
                slot_id = f"{col},{row}"
                key_index = row * self._cols + col
                mapping[slot_id] = key_index
        return mapping

    def _get_device_id(self) -> str:
        """Get unique device identifier."""
        if self._device_id is None:
            try:
                # Try to get serial number from opened device
                serial = self._device.get_serial_number()
                if serial:
                    self._device_id = serial
                else:
                    # Fallback to HID identifier
                    self._device_id = self.hid
            except Exception:
                # Fallback to HID identifier
                self._device_id = self.hid
        return self._device_id

    @property
    def id(self) -> str:
        """This is a hardware identifier for the device. It is unique for the device and does not change."""
        return self._get_device_id()

    @property
    def hid(self) -> str:
        """Return HID identifier string in format vendor_id:product_id:serial."""
        if self._hid is None:
            try:
                vendor_id = self._device.vendor_id()
                product_id = self._device.product_id()
                try:
                    serial = self._device.get_serial_number() or "unknown"
                except Exception:
                    serial = "unknown"
                self._hid = f"{vendor_id:04X}:{product_id:04X}:{serial}"
            except Exception as e:
                logger.warning(f"Could not get HID info: {e}")
                self._hid = "0000:0000:unknown"
        return self._hid

    @property
    def slots(self) -> list[hw_events.HWSlot]:
        """Return list of HWSlot objects for all keys."""
        return self._slots

    async def wake_screen(self) -> None:
        """Wake the screen."""
        # StreamDeck doesn't have explicit wake command, but we can set brightness
        await self.set_brightness(100)

    async def sleep_screen(self) -> None:
        """Sleep the screen."""
        # StreamDeck doesn't have explicit sleep command, but we can set brightness to 0
        await self.set_brightness(0)

    async def clear_key(self, target: int = 0xFF) -> None:
        """Clear a key, or all keys if target is 0xFF."""
        if target == 0xFF:
            # Clear all keys
            for key_index in range(self._rows * self._cols):
                await anyio.to_thread.run_sync(
                    self._device.set_key_image, key_index, None
                )
        else:
            await anyio.to_thread.run_sync(self._device.set_key_image, target, None)

    async def refresh(self) -> None:
        """Refresh the screen."""
        # StreamDeck doesn't have explicit refresh command
        # The display updates automatically when images are set
        pass

    async def set_brightness(self, value: int) -> None:
        """Set screen brightness.

        Args:
            value: Brightness percentage (0-100)
        """
        await anyio.to_thread.run_sync(self._device.set_brightness, value)

    async def set_image(self, slot_id: str, image: bytes) -> None:
        """Set a key image (HWDevice protocol method).

        Args:
            slot_id: Slot identifier (e.g., "0,0")
            image: Image bytes
        """
        await self.set_key_image(slot_id, image)

    async def set_key_image(self, slot_id: str, image: bytes) -> None:
        """Set a key image.

        Args:
            slot_id: Slot identifier (e.g., "0,0")
            image: Image bytes
        """
        if self._disconnected:
            return

        key_index = self._slot_to_key_map.get(slot_id)
        if key_index is None:
            logger.error(f"Slot {slot_id} not found")
            return
        try:
            await anyio.to_thread.run_sync(self._device.set_key_image, key_index, image)
        except Exception as e:
            # Check if this is a device disconnection error
            error_msg = str(e)
            error_type = type(e).__name__
            if (
                "Failed to write" in error_msg
                or "TransportError" in error_type
                or "No HID device" in error_msg
            ):
                logger.warning(f"Device write failed (likely disconnected): {e}")
                self._disconnected = True
                # Close the device to trigger disconnection handling in launch_device context manager
                try:
                    await anyio.to_thread.run_sync(self._device.close)
                except Exception:
                    pass
                # Close the event stream to signal disconnection to subscribe loop
                try:
                    await self._event_send.aclose()
                except Exception:
                    pass
                # Return silently - the device_loop will handle disconnection naturally
                return
            raise

    async def clear_slot(self, slot_id: str) -> None:
        """Clear a slot (HWDevice protocol method).

        Args:
            slot_id: Slot identifier (e.g., "0,0")
        """
        if self._disconnected:
            return

        key_index = self._slot_to_key_map.get(slot_id)
        if key_index is None:
            logger.error(f"Slot {slot_id} not found")
            return
        try:
            await anyio.to_thread.run_sync(self._device.set_key_image, key_index, None)
        except Exception as e:
            # Check if this is a device disconnection error
            error_msg = str(e)
            error_type = type(e).__name__
            if (
                "Failed to write" in error_msg
                or "TransportError" in error_type
                or "No HID device" in error_msg
            ):
                logger.warning(f"Device write failed (likely disconnected): {e}")
                self._disconnected = True
                # Close the device to trigger disconnection handling in launch_device context manager
                try:
                    await anyio.to_thread.run_sync(self._device.close)
                except Exception:
                    pass
                # Close the event stream to signal disconnection to subscribe loop
                try:
                    await self._event_send.aclose()
                except Exception:
                    pass
                # Return silently - the device_loop will handle disconnection naturally
                return
            raise

    async def subscribe(self) -> AsyncIterator[hw_events.HardwareEvent]:
        """Subscribe to hardware events from the device."""
        # The device callback will send events to _event_send
        # We need to detect if the device disconnects
        try:
            async for event in self._event_receive:
                yield event
        except (anyio.BrokenResourceError, anyio.ClosedResourceError):
            # Stream was closed (device disconnected), exit gracefully
            logger.debug("Subscribe loop ended: device disconnected")
            return
        except Exception as e:
            # Other exceptions should be logged and re-raised
            logger.debug(f"Subscribe loop ended with error: {e}")
            raise

    async def _on_key_event(self, device: StreamDeck, key: int, state: bool) -> None:
        """Handle a key event from the StreamDeck device."""
        try:
            slot_id = self._key_to_slot_map.get(key)
            if slot_id is None:
                logger.warning(f"Slot not found for key: {key}")
                return

            device_id = self.id
            if state:
                event = hw_events.KeyDownEvent(device_id=device_id, key_id=slot_id)
            else:
                event = hw_events.KeyUpEvent(device_id=device_id, key_id=slot_id)

            await self._event_send.send(event)

        except Exception as e:
            logger.exception(f"Error handling key event: {e}", exc_info=True)


@asynccontextmanager
async def launch_device(device: StreamDeck):
    """Launch and manage a StreamDeck device.

    Args:
        device: StreamDeck device object from DeviceManager.enumerate()
    """
    dock_device = ElgatoDockDevice(device)

    try:
        # Open the device
        await anyio.to_thread.run_sync(device.open)
        await anyio.to_thread.run_sync(device._reset_key_stream)

        # Set up key callback
        device.set_key_callback_async(dock_device._on_key_event)

        logger.info(f"Device opened: {dock_device.id}")

        yield dock_device

    except Exception as e:
        logger.error(f"Error launching device: {e}", exc_info=True)
        yield None
        return

    finally:
        try:
            logger.info(f"Stopping device: {dock_device.id}")
            await anyio.to_thread.run_sync(device.reset)
            await anyio.to_thread.run_sync(device.close)
            logger.info("Device closed")
        except Exception as e:
            logger.error(f"Error closing device: {e}", exc_info=True)
