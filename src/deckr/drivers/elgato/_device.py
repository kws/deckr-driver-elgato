"""Elgato Stream Deck device implementation."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import anyio
from deckr.hardware.descriptors import (
    DECKR_DEVICE_POWER,
    DECKR_INPUT_BUTTON,
    DECKR_OUTPUT_RASTER,
    CapabilityDescriptor,
    CapabilityRef,
    CapabilitySchema,
    ControlDescriptor,
    ControlGeometry,
    DeviceConnection,
    DeviceDescriptor,
    DeviceIdentifier,
    DeviceSourceReference,
)
from StreamDeck.Devices.StreamDeck import StreamDeck

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ControlInputEvent:
    control_id: str
    capability_id: str
    event_type: str
    value: dict[str, Any]


def _button_value_schema(events: tuple[str, ...], schema_id: str) -> CapabilitySchema:
    return CapabilitySchema.model_validate(
        {
            "schemaId": schema_id,
            "schema": {
                "type": "object",
                "required": ["eventType"],
                "properties": {"eventType": {"enum": list(events)}},
                "additionalProperties": False,
            },
        }
    )


def _raster_command_schema(width: int, height: int) -> CapabilitySchema:
    return CapabilitySchema.model_validate(
        {
            "schemaId": "deckr.command.output.raster.bitmap.v1",
            "schema": {
                "type": "object",
                "required": ["commandType"],
                "properties": {
                    "commandType": {"enum": ["set_frame", "clear"]},
                    "image": {"type": "string", "contentEncoding": "base64"},
                    "encoding": {"enum": ["jpeg", "png"]},
                    "width": {"const": width},
                    "height": {"const": height},
                },
                "additionalProperties": False,
            },
        }
    )


def _momentary_button_capability() -> CapabilityDescriptor:
    return CapabilityDescriptor(
        capabilityId="button.momentary",
        family=DECKR_INPUT_BUTTON,
        type="momentary",
        direction="input",
        access=("emits",),
        valueSchema=_button_value_schema(
            ("down", "up"),
            "deckr.value.input.button.momentary.v1",
        ),
        eventTypes=("down", "up"),
    )


def _activation_button_capability(control_id: str) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        capabilityId="button.press",
        family=DECKR_INPUT_BUTTON,
        type="activation",
        direction="input",
        access=("emits",),
        valueSchema=_button_value_schema(
            ("press",),
            "deckr.value.input.button.activation.v1",
        ),
        eventTypes=("press",),
        projection={
            "owner": "hardware_manager",
            "source": {
                "controlId": control_id,
                "capabilityId": "button.momentary",
            },
        },
    )


def _raster_capability(width: int, height: int, rotation: int) -> CapabilityDescriptor:
    return CapabilityDescriptor.model_validate(
        {
            "capabilityId": "raster.bitmap",
            "family": DECKR_OUTPUT_RASTER,
            "type": "bitmap",
            "direction": "output",
            "access": ["settable"],
            "commandSchema": _raster_command_schema(width, height).model_dump(
                by_alias=True,
                exclude_none=True,
                mode="json",
            ),
            "commandTypes": ["set_frame", "clear"],
            "constraints": [
                {"type": "fixed", "subject": "width", "value": width, "unit": "pixel"},
                {
                    "type": "fixed",
                    "subject": "height",
                    "value": height,
                    "unit": "pixel",
                },
                {
                    "type": "fixed",
                    "subject": "rotation",
                    "value": rotation,
                    "unit": "degree",
                },
                {"type": "enum", "subject": "encoding", "values": ["jpeg", "png"]},
            ],
            "units": [
                {"subject": "width", "unit": "pixel"},
                {"subject": "height", "unit": "pixel"},
                {"subject": "rotation", "unit": "degree"},
            ],
        }
    )


def _power_capability() -> CapabilityDescriptor:
    return CapabilityDescriptor(
        capabilityId="device.power",
        family=DECKR_DEVICE_POWER,
        type="screen",
        direction="command",
        access=("invokable",),
        commandTypes=("sleep", "wake"),
    )


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
        self._slot_to_key_map = self._create_slot_to_key_map()
        self._key_to_slot_map = {v: k for k, v in self._slot_to_key_map.items()}
        self._event_send, self._event_receive = anyio.create_memory_object_stream[
            ControlInputEvent
        ](max_buffer_size=100)
        self._device_id = None
        self._hid = None
        self._disconnected = False

    def _create_controls(self) -> list[ControlDescriptor]:
        """Create v1 control descriptors for all keys on the device."""
        controls = []
        width = 72
        height = 72
        rotation = 180
        connection_id = "usb-hid-0"
        for row in range(self._rows):
            for col in range(self._cols):
                control_id = f"{col},{row}"
                controls.append(
                    ControlDescriptor(
                        controlId=control_id,
                        kind="bitmap_key",
                        label=f"Key {col},{row}",
                        geometry=ControlGeometry(
                            x=col,
                            y=row,
                            width=1,
                            height=1,
                            unit="grid",
                        ),
                        inputCapabilities=(
                            _momentary_button_capability(),
                            _activation_button_capability(control_id),
                        ),
                        outputCapabilities=(
                            _raster_capability(width, height, rotation),
                        ),
                        sources=(
                            DeviceSourceReference(
                                sourceId=f"key-report-{col}-{row}",
                                type="hid",
                                connectionId=connection_id,
                                facts={"keyIndex": row * self._cols + col},
                            ),
                        ),
                    )
                )
        return controls

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
    def descriptor(self) -> DeviceDescriptor:
        """Return the canonical Deckr device descriptor for this device."""
        vendor_id = None
        product_id = None
        try:
            vendor_id = self._device.vendor_id()
            product_id = self._device.product_id()
        except Exception:
            logger.debug("Could not read Elgato USB ids", exc_info=True)
        serial = self.id
        connections: list[DeviceConnection] = [
            DeviceConnection(
                connectionId="usb-hid-0",
                type="hid",
                status="connected",
                transport="usb",
                facts={
                    key: value
                    for key, value in {
                        "vendorId": vendor_id,
                        "productId": product_id,
                        "serialNumber": serial,
                    }.items()
                    if value is not None
                },
            )
        ]
        identifiers: list[DeviceIdentifier] = []
        if vendor_id is not None and product_id is not None:
            identifiers.append(
                DeviceIdentifier(
                    type="usb.vendor_product",
                    namespace="usb",
                    value=f"{vendor_id:04x}:{product_id:04x}",
                )
            )
        return DeviceDescriptor(
            deviceId=self.id,
            fingerprint=self.hid,
            displayName=getattr(self._device, "deck_type", lambda: "Stream Deck")(),
            manufacturer="Elgato",
            model=getattr(self._device, "deck_type", lambda: None)(),
            serialNumber=serial,
            identifiers=tuple(identifiers),
            connections=tuple(connections),
            defaultStatusIndicator=CapabilityRef(
                controlId="0,0",
                capabilityId="raster.bitmap",
            ),
            capabilities=(_power_capability(),),
            controls=tuple(self._create_controls()),
        )

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
        """Set a key image on the private live device.

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
        """Clear a slot on the private live device.

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

    async def subscribe(self) -> AsyncIterator[ControlInputEvent]:
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

            if state:
                event = ControlInputEvent(
                    control_id=slot_id,
                    capability_id="button.momentary",
                    event_type="down",
                    value={"eventType": "down"},
                )
                await self._event_send.send(event)
            else:
                await self._event_send.send(
                    ControlInputEvent(
                        control_id=slot_id,
                        capability_id="button.momentary",
                        event_type="up",
                        value={"eventType": "up"},
                    )
                )
                await self._event_send.send(
                    ControlInputEvent(
                        control_id=slot_id,
                        capability_id="button.press",
                        event_type="press",
                        value={"eventType": "press"},
                    )
                )

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
