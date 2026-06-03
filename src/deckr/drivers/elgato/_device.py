"""Elgato Stream Deck device adapter."""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import logging
import re
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal

import anyio
from deckr.contracts.messages import DeckrMessage
from deckr.hardware import messages as hw_messages
from deckr.hardware.capabilities import (
    RasterBitmapClearParams,
    RasterBitmapSetFrameParams,
    button_momentary_value_schema,
    device_power_command_params,
    device_power_command_schema,
    encoder_relative_value_schema,
    raster_bitmap_command_params,
    raster_bitmap_command_schema,
    touch_gesture_value_schema,
)
from deckr.hardware.descriptors import (
    DECKR_DEVICE_POWER,
    DECKR_INPUT_BUTTON,
    DECKR_INPUT_ENCODER,
    DECKR_INPUT_TOUCH,
    DECKR_OUTPUT_RASTER,
    CapabilityDescriptor,
    ControlDescriptor,
    ControlGeometry,
    DescriptorCapabilityRef,
    DeviceConnection,
    DeviceDescriptor,
    DeviceIdentifier,
    DeviceSourceReference,
)
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError
from StreamDeck.Devices.StreamDeck import (
    DialEventType,
    StreamDeck,
    TouchscreenEventType,
)
from StreamDeck.ImageHelpers import PILHelper

logger = logging.getLogger(__name__)

_IDENTITY_RE = re.compile(r"[^A-Za-z0-9._-]+")

RasterSurfaceKind = Literal["key", "screen", "touchscreen"]


@dataclass(frozen=True, slots=True)
class ControlInputEvent:
    control_id: str
    capability_id: str
    event_type: str
    value: dict[str, Any]
    sources: tuple[DeviceSourceReference, ...] = ()


@dataclass(frozen=True, slots=True)
class _RasterSurface:
    kind: RasterSurfaceKind
    width: int
    height: int
    rotation: int
    key_index: int | None = None


def _schema_dict(schema: Any) -> dict[str, Any]:
    return schema.model_dump(by_alias=True, exclude_none=True, mode="json")


def _momentary_button_capability(
    capability_id: str = "button.momentary",
) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        capabilityId=capability_id,
        family=DECKR_INPUT_BUTTON,
        type="momentary",
        direction="input",
        access=("emits",),
        valueSchema=button_momentary_value_schema(),
        eventTypes=("down", "up"),
    )


def _encoder_capability() -> CapabilityDescriptor:
    return CapabilityDescriptor(
        capabilityId="encoder.relative",
        family=DECKR_INPUT_ENCODER,
        type="relative",
        direction="input",
        access=("emits",),
        valueSchema=encoder_relative_value_schema(),
        eventTypes=("rotate",),
        constraints=(
            {
                "type": "range",
                "subject": "delta",
                "minimum": -24,
                "maximum": 24,
                "step": 1,
                "unit": "detent",
            },
        ),
        units=({"subject": "delta", "unit": "detent"},),
    )


def _touch_capability() -> CapabilityDescriptor:
    return CapabilityDescriptor(
        capabilityId="touch.gesture",
        family=DECKR_INPUT_TOUCH,
        type="gesture",
        direction="input",
        access=("emits",),
        valueSchema=touch_gesture_value_schema(),
        eventTypes=("tap", "swipe"),
    )


def _raster_capability(width: int, height: int, rotation: int) -> CapabilityDescriptor:
    return CapabilityDescriptor.model_validate(
        {
            "capabilityId": "raster.bitmap",
            "family": DECKR_OUTPUT_RASTER,
            "type": "bitmap",
            "direction": "output",
            "access": ["settable"],
            "commandSchema": _schema_dict(
                raster_bitmap_command_schema(width=width, height=height)
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
        commandSchema=device_power_command_schema(),
        commandTypes=("sleep", "wake"),
    )


def _image_format_size(image_format: Mapping[str, Any]) -> tuple[int, int] | None:
    size = image_format.get("size")
    if (
        not isinstance(size, tuple | list)
        or len(size) != 2
        or not isinstance(size[0], int)
        or not isinstance(size[1], int)
        or size[0] <= 0
        or size[1] <= 0
    ):
        return None
    return int(size[0]), int(size[1])


def _image_format_rotation(image_format: Mapping[str, Any]) -> int:
    rotation = image_format.get("rotation", 0)
    return int(rotation) if isinstance(rotation, int | float) else 0


def _image_format_is_supported(image_format: Mapping[str, Any]) -> bool:
    return bool(_image_format_size(image_format) and image_format.get("format"))


def _identity_token(value: str, *, fallback_prefix: str) -> str:
    normalized = _IDENTITY_RE.sub("-", value.strip()).strip("-")
    if normalized:
        return normalized
    digest = hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()[:16]
    return f"{fallback_prefix}-{digest}"


def _hash_token(value: str, *, prefix: str) -> str:
    return f"{prefix}-{hashlib.sha256(value.encode()).hexdigest()[:16]}"


def _call_text(obj: Any, name: str) -> str | None:
    func = getattr(obj, name, None)
    if func is None:
        return None
    try:
        value = func()
    except Exception:
        logger.debug("Could not read StreamDeck %s", name, exc_info=True)
        return None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _call_int(obj: Any, name: str) -> int | None:
    func = getattr(obj, name, None)
    if func is None:
        return None
    try:
        value = func()
    except Exception:
        logger.debug("Could not read StreamDeck %s", name, exc_info=True)
        return None
    return int(value) if isinstance(value, int) else None


def _transport_path(device: StreamDeck) -> str | None:
    transport_device = getattr(device, "device", None)
    path = getattr(transport_device, "path", None)
    if path is None:
        return None
    try:
        value = path()
    except Exception:
        logger.debug("Could not read StreamDeck transport path", exc_info=True)
        return None
    return str(value) if value is not None else None


def _looks_like_disconnect(exc: BaseException) -> bool:
    message = str(exc)
    name = type(exc).__name__
    return (
        "Failed to write" in message
        or "Failed to read" in message
        or "No HID device" in message
        or "TransportError" in name
        or "not open" in message
    )


class ElgatoDockDevice:
    """Runtime adapter for one opened StreamDeck device."""

    def __init__(self, device: StreamDeck):
        self._device = device
        self._rows, self._cols = device.key_layout()
        self._event_send, self._event_receive = anyio.create_memory_object_stream[
            ControlInputEvent
        ](max_buffer_size=100)
        self._device_id: str | None = None
        self._fingerprint: str | None = None
        self._descriptor: DeviceDescriptor | None = None
        self._disconnected = False

        self._key_controls: dict[int, str] = {}
        self._dial_controls: dict[int, str] = {}
        self._raster_surfaces: dict[str, _RasterSurface] = {}
        self._configure_control_maps()

    @property
    def raw_device(self) -> StreamDeck:
        return self._device

    @property
    def id(self) -> str:
        if self._device_id is None:
            serial = _call_text(self._device, "get_serial_number")
            if serial:
                self._device_id = _identity_token(serial, fallback_prefix="streamdeck")
            else:
                self._device_id = _hash_token(self.fingerprint, prefix="streamdeck")
        return self._device_id

    @property
    def fingerprint(self) -> str:
        if self._fingerprint is None:
            vendor_id = _call_int(self._device, "vendor_id")
            product_id = _call_int(self._device, "product_id")
            serial = _call_text(self._device, "get_serial_number")
            path = _transport_path(self._device)
            if vendor_id is not None and product_id is not None and serial:
                self._fingerprint = f"usb:{vendor_id:04x}:{product_id:04x}:{serial}"
            elif vendor_id is not None and product_id is not None and path:
                self._fingerprint = (
                    f"usb:{vendor_id:04x}:{product_id:04x}:"
                    f"path:{hashlib.sha256(path.encode()).hexdigest()[:16]}"
                )
            elif path:
                self._fingerprint = (
                    f"usb:path:{hashlib.sha256(path.encode()).hexdigest()[:16]}"
                )
            else:
                self._fingerprint = "usb:unknown:" + hashlib.sha256(
                    str(id(self._device)).encode()
                ).hexdigest()[:16]
        return self._fingerprint

    @property
    def hid(self) -> str:
        return self.fingerprint

    @property
    def descriptor(self) -> DeviceDescriptor:
        if self._descriptor is None:
            self._descriptor = self._build_descriptor()
        return self._descriptor

    @property
    def disconnected(self) -> bool:
        return self._disconnected

    def is_connected(self) -> bool:
        try:
            connected = getattr(self._device, "connected", lambda: True)()
            open_ = getattr(self._device, "is_open", lambda: True)()
        except Exception:
            return False
        return bool(connected and open_ and not self._disconnected)

    async def mark_disconnected(self) -> None:
        if self._disconnected:
            return
        self._disconnected = True
        await self._event_send.aclose()

    async def subscribe(self) -> AsyncIterator[ControlInputEvent]:
        try:
            async for event in self._event_receive:
                yield event
        except (anyio.BrokenResourceError, anyio.ClosedResourceError):
            return

    async def handle_command(
        self,
        envelope: DeckrMessage,
        *,
        manager_id: str,
    ) -> bool | None:
        if self._disconnected:
            return False
        ref = hw_messages.hardware_device_ref_from_message(envelope)
        if ref is None or ref.manager_id != manager_id or ref.device_id != self.id:
            return None
        body = hw_messages.hardware_body_from_message(envelope)
        if not isinstance(body, hw_messages.ControlCommandMessage):
            return None
        if body.capability_id == "device.power":
            return await self._handle_power_command(body)
        if body.capability_id == "raster.bitmap":
            return await self._handle_raster_command(body)
        return None

    async def reset_outputs(self) -> None:
        for control_id in sorted(self._raster_surfaces):
            await self._clear_raster(control_id)

    async def _handle_power_command(
        self,
        body: hw_messages.ControlCommandMessage,
    ) -> bool | None:
        try:
            device_power_command_params(body.params)
        except ValidationError as exc:
            logger.warning("Ignoring invalid Elgato power command params: %s", exc)
            return None
        if body.command_type == "wake":
            await self._run_device_call(self._device.set_brightness, 100)
            return True
        if body.command_type == "sleep":
            await self._run_device_call(self._device.set_brightness, 0)
            return True
        return None

    async def _handle_raster_command(
        self,
        body: hw_messages.ControlCommandMessage,
    ) -> bool | None:
        if body.control_id is None:
            return None
        _ = self.descriptor
        if body.control_id not in self._raster_surfaces:
            return None
        try:
            params = raster_bitmap_command_params(body.command_type, body.params)
        except (ValueError, ValidationError) as exc:
            logger.warning("Ignoring invalid Elgato raster command params: %s", exc)
            return None
        if isinstance(params, RasterBitmapClearParams):
            await self._clear_raster(body.control_id)
            return True
        if isinstance(params, RasterBitmapSetFrameParams):
            try:
                image = _decode_raster_image(params.image)
            except (ValueError, binascii.Error, UnidentifiedImageError) as exc:
                logger.warning("Ignoring invalid Elgato raster image payload: %s", exc)
                return None
            await self._set_raster(body.control_id, image)
            return True
        return None

    async def _set_raster(self, control_id: str, image: Image.Image) -> None:
        surface = self._raster_surfaces[control_id]
        image = _fit_rgb_canvas(image, (surface.width, surface.height))
        if surface.kind == "key":
            native = PILHelper.to_native_key_format(self._device, image)
            await self._run_device_call(
                self._device.set_key_image,
                surface.key_index,
                native,
            )
            return
        if surface.kind == "screen":
            native = PILHelper.to_native_screen_format(self._device, image)
            await self._run_device_call(self._device.set_screen_image, native)
            return
        native = PILHelper.to_native_touchscreen_format(self._device, image)
        await self._run_device_call(
            self._device.set_touchscreen_image,
            native,
            0,
            0,
            surface.width,
            surface.height,
        )

    async def _clear_raster(self, control_id: str) -> None:
        if self._disconnected:
            return
        surface = self._raster_surfaces.get(control_id)
        if surface is None:
            return
        if surface.kind == "key":
            await self._run_device_call(
                self._device.set_key_image,
                surface.key_index,
                None,
            )
        elif surface.kind == "screen":
            await self._run_device_call(self._device.set_screen_image, None)
        else:
            await self._run_device_call(self._device.set_touchscreen_image, None)

    async def _run_device_call(self, func, *args) -> Any:
        if self._disconnected:
            return None
        try:
            return await anyio.to_thread.run_sync(func, *args)
        except Exception as exc:
            if _looks_like_disconnect(exc):
                logger.warning("Elgato device disconnected during I/O: %s", exc)
                await self.mark_disconnected()
                try:
                    await anyio.to_thread.run_sync(self._device.close)
                except Exception:
                    pass
                return None
            raise

    async def _on_key_event(
        self,
        _device: StreamDeck,
        key_index: int,
        pressed: bool,
    ) -> None:
        control_id = self._key_controls.get(key_index)
        if control_id is None:
            logger.debug("Ignoring unknown Elgato key index %s", key_index)
            return
        event_type = "down" if pressed else "up"
        await self._send_input(
            ControlInputEvent(
                control_id=control_id,
                capability_id="button.momentary",
                event_type=event_type,
                value={"eventType": event_type},
                sources=(
                    DeviceSourceReference(
                        sourceId=f"key-report-{key_index}",
                        type="hid",
                        connectionId="usb-hid-0",
                        facts={"keyIndex": key_index},
                    ),
                ),
            )
        )

    async def _on_dial_event(
        self,
        _device: StreamDeck,
        dial_index: int,
        event_type: DialEventType,
        value: int | bool,
    ) -> None:
        control_id = self._dial_controls.get(dial_index)
        if control_id is None:
            logger.debug("Ignoring unknown Elgato dial index %s", dial_index)
            return
        if event_type == DialEventType.TURN:
            delta = int(value)
            if delta == 0:
                return
            await self._send_input(
                ControlInputEvent(
                    control_id=control_id,
                    capability_id="encoder.relative",
                    event_type="rotate",
                    value={
                        "delta": delta,
                        "direction": (
                            "clockwise" if delta > 0 else "counterclockwise"
                        ),
                    },
                    sources=(
                        DeviceSourceReference(
                            sourceId=f"dial-report-{dial_index}",
                            type="hid",
                            connectionId="usb-hid-0",
                            facts={"dialIndex": dial_index},
                        ),
                    ),
                )
            )
            return
        if event_type == DialEventType.PUSH:
            button_event = "down" if bool(value) else "up"
            await self._send_input(
                ControlInputEvent(
                    control_id=control_id,
                    capability_id="button.momentary",
                    event_type=button_event,
                    value={"eventType": button_event},
                    sources=(
                        DeviceSourceReference(
                            sourceId=f"dial-button-report-{dial_index}",
                            type="hid",
                            connectionId="usb-hid-0",
                            facts={"dialIndex": dial_index},
                        ),
                    ),
                )
            )

    async def _on_touchscreen_event(
        self,
        _device: StreamDeck,
        event_type: TouchscreenEventType,
        value: Mapping[str, Any],
    ) -> None:
        if "touchscreen.0" not in self._raster_surfaces:
            return
        if event_type == TouchscreenEventType.SHORT:
            await self._send_input(
                ControlInputEvent(
                    control_id="touchscreen.0",
                    capability_id="touch.gesture",
                    event_type="tap",
                    value={"eventType": "tap"},
                    sources=(
                        DeviceSourceReference(
                            sourceId="touchscreen-report-0",
                            type="hid",
                            connectionId="usb-hid-0",
                            facts=dict(value),
                        ),
                    ),
                )
            )
            return
        if event_type != TouchscreenEventType.DRAG:
            return
        x_start = _int_value(value.get("x"))
        x_end = _int_value(value.get("x_out"))
        if x_start is None or x_end is None or x_start == x_end:
            return
        await self._send_input(
            ControlInputEvent(
                control_id="touchscreen.0",
                capability_id="touch.gesture",
                event_type="swipe",
                value={
                    "eventType": "swipe",
                    "direction": "right" if x_end > x_start else "left",
                },
                sources=(
                    DeviceSourceReference(
                        sourceId="touchscreen-report-0",
                        type="hid",
                        connectionId="usb-hid-0",
                        facts=dict(value),
                    ),
                ),
            )
        )

    async def _send_input(self, event: ControlInputEvent) -> None:
        if self._disconnected:
            return
        try:
            await self._event_send.send(event)
        except (anyio.BrokenResourceError, anyio.ClosedResourceError):
            self._disconnected = True

    def _configure_control_maps(self) -> None:
        visual_key_count = self._visual_key_count()
        key_count = self._total_key_count()
        for key_index in range(key_count):
            if key_index < visual_key_count:
                row = key_index // self._cols if self._cols else 0
                col = key_index % self._cols if self._cols else key_index
                self._key_controls[key_index] = f"key.{col}.{row}"
            else:
                self._key_controls[key_index] = f"button.{key_index - visual_key_count}"
        for dial_index in range(self._dial_count()):
            self._dial_controls[dial_index] = f"dial.{dial_index}"

    def _build_descriptor(self) -> DeviceDescriptor:
        vendor_id = _call_int(self._device, "vendor_id")
        product_id = _call_int(self._device, "product_id")
        serial = _call_text(self._device, "get_serial_number")
        firmware = _call_text(self._device, "get_firmware_version")
        path = _transport_path(self._device)
        model = _call_text(self._device, "deck_type") or "Stream Deck"

        facts = {
            key: value
            for key, value in {
                "vendorId": vendor_id,
                "productId": product_id,
                "serialNumber": serial,
                "firmwareVersion": firmware,
                "path": path,
            }.items()
            if value is not None
        }
        identifiers: list[DeviceIdentifier] = []
        if vendor_id is not None and product_id is not None:
            identifiers.append(
                DeviceIdentifier(
                    type="usb.vendor_product",
                    namespace="usb",
                    value=f"{vendor_id:04x}:{product_id:04x}",
                )
            )
        if serial:
            identifiers.append(
                DeviceIdentifier(
                    type="serial",
                    namespace="elgato",
                    value=serial,
                )
            )

        controls = tuple(self._controls())
        default_status = _default_status_indicator(controls)
        capabilities: list[CapabilityDescriptor] = []
        if self._supports_power():
            capabilities.append(_power_capability())

        return DeviceDescriptor(
            deviceId=self.id,
            fingerprint=self.fingerprint,
            displayName=model,
            manufacturer="Elgato",
            model=model,
            modelId=(
                f"{vendor_id:04x}:{product_id:04x}"
                if vendor_id is not None and product_id is not None
                else None
            ),
            serialNumber=serial,
            firmwareVersion=firmware,
            identifiers=tuple(identifiers),
            connections=(
                DeviceConnection(
                    connectionId="usb-hid-0",
                    type="hid",
                    status="connected",
                    transport="usb",
                    facts=facts,
                ),
            ),
            defaultStatusIndicator=default_status,
            capabilities=tuple(capabilities),
            controls=controls,
        )

    def _controls(self) -> list[ControlDescriptor]:
        controls: list[ControlDescriptor] = []
        controls.extend(self._key_controls_for_descriptor())
        controls.extend(self._dial_controls_for_descriptor())
        controls.extend(self._screen_controls_for_descriptor())
        return controls

    def _key_controls_for_descriptor(self) -> list[ControlDescriptor]:
        controls: list[ControlDescriptor] = []
        visual_key_count = self._visual_key_count()
        key_count = self._total_key_count()
        key_image = self._safe_image_format("key_image_format")
        key_size = _image_format_size(key_image or {})
        key_rotation = _image_format_rotation(key_image or {})
        for key_index in range(key_count):
            control_id = self._key_controls[key_index]
            row = key_index // self._cols if self._cols else 0
            col = key_index % self._cols if self._cols else key_index
            is_visual = key_index < visual_key_count and key_size is not None
            output_capabilities: tuple[CapabilityDescriptor, ...] = ()
            if is_visual:
                width, height = key_size
                output_capabilities = (_raster_capability(width, height, key_rotation),)
                self._raster_surfaces[control_id] = _RasterSurface(
                    kind="key",
                    width=width,
                    height=height,
                    rotation=key_rotation,
                    key_index=key_index,
                )
            controls.append(
                ControlDescriptor(
                    controlId=control_id,
                    kind="bitmap_key" if is_visual else "button",
                    label=(
                        f"Key {col},{row}"
                        if is_visual
                        else f"Button {key_index - visual_key_count}"
                    ),
                    groupId="key-grid" if is_visual else "hardware-buttons",
                    surfaceId="key-grid" if is_visual else None,
                    geometry=ControlGeometry(
                        x=col,
                        y=row,
                        width=1,
                        height=1,
                        unit="grid",
                    ),
                    inputCapabilities=(_momentary_button_capability(),),
                    outputCapabilities=output_capabilities,
                    sources=(
                        DeviceSourceReference(
                            sourceId=f"key-report-{key_index}",
                            type="hid",
                            connectionId="usb-hid-0",
                            facts={"keyIndex": key_index},
                        ),
                    ),
                )
            )
        return controls

    def _dial_controls_for_descriptor(self) -> list[ControlDescriptor]:
        controls: list[ControlDescriptor] = []
        base_y = self._rows + 1
        for dial_index in range(self._dial_count()):
            controls.append(
                ControlDescriptor(
                    controlId=f"dial.{dial_index}",
                    kind="rotary_encoder",
                    label=f"Dial {dial_index + 1}",
                    groupId="dial-strip",
                    geometry=ControlGeometry(
                        x=dial_index,
                        y=base_y,
                        width=1,
                        height=1,
                        unit="grid",
                    ),
                    inputCapabilities=(
                        _encoder_capability(),
                        _momentary_button_capability(),
                    ),
                    sources=(
                        DeviceSourceReference(
                            sourceId=f"dial-report-{dial_index}",
                            type="hid",
                            connectionId="usb-hid-0",
                            facts={"dialIndex": dial_index},
                        ),
                    ),
                )
            )
        return controls

    def _screen_controls_for_descriptor(self) -> list[ControlDescriptor]:
        controls: list[ControlDescriptor] = []
        touchscreen = self._safe_image_format("touchscreen_image_format")
        touchscreen_size = _image_format_size(touchscreen or {})
        if touchscreen_size is not None and _image_format_is_supported(touchscreen or {}):
            width, height = touchscreen_size
            rotation = _image_format_rotation(touchscreen or {})
            self._raster_surfaces["touchscreen.0"] = _RasterSurface(
                kind="touchscreen",
                width=width,
                height=height,
                rotation=rotation,
            )
            controls.append(
                ControlDescriptor(
                    controlId="touchscreen.0",
                    kind="touch_surface",
                    label="Touchscreen",
                    groupId="touchscreen",
                    surfaceId="touchscreen",
                    geometry=ControlGeometry(
                        x=0,
                        y=self._rows,
                        width=max(self._cols, 1),
                        height=1,
                        unit="grid",
                    ),
                    inputCapabilities=(_touch_capability(),),
                    outputCapabilities=(_raster_capability(width, height, rotation),),
                    sources=(
                        DeviceSourceReference(
                            sourceId="touchscreen-report-0",
                            type="hid",
                            connectionId="usb-hid-0",
                        ),
                    ),
                )
            )

        screen = self._safe_image_format("screen_image_format")
        screen_size = _image_format_size(screen or {})
        if screen_size is not None and _image_format_is_supported(screen or {}):
            width, height = screen_size
            rotation = _image_format_rotation(screen or {})
            self._raster_surfaces["screen.0"] = _RasterSurface(
                kind="screen",
                width=width,
                height=height,
                rotation=rotation,
            )
            controls.append(
                ControlDescriptor(
                    controlId="screen.0",
                    kind="display",
                    label="Screen",
                    groupId="screen",
                    surfaceId="screen",
                    geometry=ControlGeometry(
                        x=0,
                        y=self._rows,
                        width=max(self._cols, 1),
                        height=1,
                        unit="grid",
                    ),
                    outputCapabilities=(_raster_capability(width, height, rotation),),
                    sources=(
                        DeviceSourceReference(
                            sourceId="screen-output-0",
                            type="hid",
                            connectionId="usb-hid-0",
                        ),
                    ),
                )
            )
        return controls

    def _safe_image_format(self, method_name: str) -> Mapping[str, Any] | None:
        method = getattr(self._device, method_name, None)
        if method is None:
            return None
        try:
            value = method()
        except Exception:
            logger.debug("Could not read StreamDeck %s", method_name, exc_info=True)
            return None
        return dict(value) if isinstance(value, Mapping) else None

    def _supports_power(self) -> bool:
        return bool(getattr(self._device, "is_visual", lambda: False)())

    def _physical_key_count(self) -> int:
        try:
            return int(self._device.key_count())
        except Exception:
            return max(self._rows * self._cols, 0)

    def _total_key_count(self) -> int:
        try:
            touch_keys = int(self._device.touch_key_count())
        except Exception:
            touch_keys = 0
        return self._physical_key_count() + max(touch_keys, 0)

    def _visual_key_count(self) -> int:
        if not bool(getattr(self._device, "is_visual", lambda: False)()):
            return 0
        return min(self._physical_key_count(), max(self._rows * self._cols, 0))

    def _dial_count(self) -> int:
        try:
            return int(self._device.dial_count())
        except Exception:
            return 0


def _decode_raster_image(encoded: str) -> Image.Image:
    raw = base64.b64decode(encoded, validate=True)
    image = Image.open(io.BytesIO(raw))
    image.load()
    return image


def _fit_rgb_canvas(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    source = image.convert("RGBA")
    source.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "black")
    left = (size[0] - source.width) // 2
    top = (size[1] - source.height) // 2
    canvas.paste(source, (left, top), source)
    return canvas


def _default_status_indicator(
    controls: tuple[ControlDescriptor, ...],
) -> DescriptorCapabilityRef | None:
    for control in controls:
        for capability in control.output_capabilities:
            if capability.capability_id == "raster.bitmap":
                return DescriptorCapabilityRef(
                    controlId=control.control_id,
                    capabilityId=capability.capability_id,
                )
    return None


def _int_value(value: Any) -> int | None:
    return int(value) if isinstance(value, int) else None


@asynccontextmanager
async def launch_device(device: StreamDeck) -> AsyncIterator[ElgatoDockDevice | None]:
    dock_device = ElgatoDockDevice(device)
    try:
        await anyio.to_thread.run_sync(device.open)
        device.set_key_callback_async(dock_device._on_key_event)
        if getattr(device, "dial_count", lambda: 0)():
            device.set_dial_callback_async(dock_device._on_dial_event)
        if getattr(device, "is_touch", lambda: False)():
            device.set_touchscreen_callback_async(dock_device._on_touchscreen_event)
        logger.info("Elgato device opened: %s", dock_device.id)
        yield dock_device
    except Exception as exc:
        logger.warning("Could not open Elgato StreamDeck device: %s", exc, exc_info=True)
        yield None
    finally:
        try:
            if hasattr(device, "set_key_callback"):
                device.set_key_callback(None)
            if hasattr(device, "set_dial_callback"):
                device.set_dial_callback(None)
            if hasattr(device, "set_touchscreen_callback"):
                device.set_touchscreen_callback(None)
        except Exception:
            logger.debug("Could not clear Elgato callbacks", exc_info=True)
        try:
            await anyio.to_thread.run_sync(device.reset)
        except Exception:
            logger.debug("Could not reset Elgato device during close", exc_info=True)
        try:
            await anyio.to_thread.run_sync(device.close)
        except Exception:
            logger.debug("Could not close Elgato device", exc_info=True)
        await dock_device.mark_disconnected()
