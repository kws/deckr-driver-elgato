from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import anyio
import deckr.hardware.messages as hw_messages
import pytest
from deckr.components import (
    ReadinessState,
    resolve_component_host_plan,
    start_components,
)
from deckr.concord import ContractValidityStatus
from deckr.contracts.messages import controller_address, hardware_manager_address
from deckr.core.config import ConfigDocument
from deckr.hardware import (
    HARDWARE_CLAIM_PROFILE_ID,
    HARDWARE_FEATURE_ID,
    DeviceRef,
    HardwareBeaconPayload,
    HardwareClaimDevice,
    HardwareClaimTerms,
    HardwareManagerRuntime,
)
from message_bus_mocks import mock_deckr
from PIL import Image

from deckr.drivers.elgato import _factory as factory_module
from deckr.drivers.elgato._device import ElgatoDockDevice
from deckr.drivers.elgato._discovery import ElgatoDeviceSupervisor

pytestmark = pytest.mark.asyncio


class _Transport:
    def __init__(self, path: str) -> None:
        self._path = path

    def path(self) -> str:
        return self._path


class _FakeStreamDeck:
    def __init__(
        self,
        *,
        serial: str,
        path: str,
        deck_type: str = "Stream Deck Mini",
        rows: int = 2,
        cols: int = 3,
        key_count: int = 6,
        touch_key_count: int = 0,
        dial_count: int = 0,
        visual: bool = True,
        touch: bool = False,
        key_size: tuple[int, int] = (80, 80),
        key_format: str = "JPEG",
        touchscreen_size: tuple[int, int] | None = None,
        screen_size: tuple[int, int] | None = None,
    ) -> None:
        self.device = _Transport(path)
        self.serial = serial
        self._deck_type = deck_type
        self._rows = rows
        self._cols = cols
        self._key_count = key_count
        self._touch_key_count = touch_key_count
        self._dial_count = dial_count
        self._visual = visual
        self._touch = touch
        self._key_size = key_size
        self._key_format = key_format
        self._touchscreen_size = touchscreen_size
        self._screen_size = screen_size
        self.opened = False
        self.closed = False
        self.key_callback = None
        self.dial_callback = None
        self.touchscreen_callback = None
        self.key_images: dict[int, bytes | None] = {}
        self.screen_image: bytes | None = None
        self.touchscreen_image: bytes | None = None
        self.brightness_values: list[int] = []
        self.reset_count = 0

    def open(self) -> None:
        self.opened = True
        self.closed = False

    def close(self) -> None:
        self.closed = True
        self.opened = False

    def reset(self) -> None:
        self.reset_count += 1

    def connected(self) -> bool:
        return self.opened and not self.closed

    def is_open(self) -> bool:
        return self.opened and not self.closed

    def key_layout(self) -> tuple[int, int]:
        return self._rows, self._cols

    def key_count(self) -> int:
        return self._key_count

    def touch_key_count(self) -> int:
        return self._touch_key_count

    def dial_count(self) -> int:
        return self._dial_count

    def is_visual(self) -> bool:
        return self._visual

    def is_touch(self) -> bool:
        return self._touch

    def deck_type(self) -> str:
        return self._deck_type

    def vendor_id(self) -> int:
        return 0x0FD9

    def product_id(self) -> int:
        return 0x0063

    def get_serial_number(self) -> str:
        return self.serial

    def get_firmware_version(self) -> str:
        return "1.0"

    def key_image_format(self) -> dict[str, Any]:
        return {
            "size": self._key_size,
            "format": self._key_format,
            "flip": (False, False),
            "rotation": 0,
        }

    def touchscreen_image_format(self) -> dict[str, Any]:
        size = self._touchscreen_size or (0, 0)
        return {
            "size": size,
            "format": "JPEG" if self._touchscreen_size else "",
            "flip": (False, False),
            "rotation": 0,
        }

    def screen_image_format(self) -> dict[str, Any]:
        size = self._screen_size or (0, 0)
        return {
            "size": size,
            "format": "JPEG" if self._screen_size else "",
            "flip": (False, False),
            "rotation": 0,
        }

    def set_key_callback_async(self, callback) -> None:
        self.key_callback = callback

    def set_key_callback(self, callback) -> None:
        self.key_callback = callback

    def set_dial_callback_async(self, callback) -> None:
        self.dial_callback = callback

    def set_dial_callback(self, callback) -> None:
        self.dial_callback = callback

    def set_touchscreen_callback_async(self, callback) -> None:
        self.touchscreen_callback = callback

    def set_touchscreen_callback(self, callback) -> None:
        self.touchscreen_callback = callback

    def set_key_image(self, key: int, image: bytes | None) -> None:
        self.key_images[key] = image

    def set_screen_image(self, image: bytes | None) -> None:
        self.screen_image = image

    def set_touchscreen_image(
        self,
        image: bytes | None,
        x_pos: int = 0,
        y_pos: int = 0,
        width: int = 0,
        height: int = 0,
    ) -> None:
        del x_pos, y_pos, width, height
        self.touchscreen_image = image

    def set_brightness(self, value: int) -> None:
        self.brightness_values.append(value)


class _FakeDeviceManager:
    def __init__(self, devices: list[_FakeStreamDeck]) -> None:
        self._devices = devices

    def enumerate(self) -> list[_FakeStreamDeck]:
        return self._devices


def _document(raw: dict[str, Any]) -> ConfigDocument:
    return ConfigDocument(raw=raw, source_path=None, base_dir=Path.cwd())


async def _wait_for_readiness(manager, runtime_name: str) -> None:
    with anyio.fail_after(3):
        while True:
            status = manager.get_component_status(runtime_name)
            if status is not None and status.readiness_state == ReadinessState.READY:
                return
            await anyio.sleep(0.01)


async def _wait_for_hardware_payload(
    deckr,
    *,
    device_ids: set[str] | None = None,
) -> HardwareBeaconPayload:
    with anyio.fail_after(3):
        while True:
            candidates = deckr.beacon.candidates(HARDWARE_FEATURE_ID)
            if candidates:
                payload = HardwareBeaconPayload.model_validate(
                    candidates[0].advertisement.payload
                )
                if device_ids is None or set(payload.devices) == device_ids:
                    return payload
            await anyio.sleep(0.01)


def _component_document() -> ConfigDocument:
    return _document(
        {
            "deckr": {
                "components": {
                    "instances": {
                        "elgato": {
                            "component": "dev.deckr.hardware.elgato",
                            "instance_id": "main",
                            "endpoints": {"hardware_manager": "elgato-main"},
                            "config": {"labels": {"room": "office"}},
                        }
                    }
                }
            }
        }
    )


def _png_payload(size: tuple[int, int] = (80, 80)) -> str:
    image = Image.new("RGBA", size, (255, 0, 0, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


async def _claim(
    runtime: HardwareManagerRuntime,
    deckr,
    controller_endpoint,
    *,
    device_id: str | None = None,
):
    descriptor = (
        runtime._devices[device_id]
        if device_id is not None
        else next(iter(runtime._devices.values()))
    )
    terms = HardwareClaimTerms(
        claimId="claim-1",
        controllerEndpoint=controller_endpoint.address,
        managerEndpoint=hardware_manager_address("manager-main"),
        devices=(
            HardwareClaimDevice(
                deviceRef=DeviceRef(
                    managerId="manager-main",
                    deviceId=descriptor.device_id,
                    fingerprint=descriptor.fingerprint,
                ),
                instanceCount=1,
            ),
        ),
    )
    contract = await deckr.concord._create_contract(
        (controller_endpoint.address, hardware_manager_address("manager-main")),
        contract_id="claim-1",
        profile=HARDWARE_CLAIM_PROFILE_ID,
        terms=terms,
        created_by=controller_endpoint.address,
    )
    await deckr.concord._attach(
        contract,
        controller_endpoint.address,
        controller_endpoint.session_id,
    )
    await runtime._reconcile_claims(reason="test")
    return contract


async def test_component_uses_hosted_context_and_managed_protocols(monkeypatch):
    instances = []

    class FakeSupervisor:
        def __init__(self, *, runtime, manager_id):
            self.runtime = runtime
            self.manager_id = manager_id
            self.started = False
            self.stopped = False
            instances.append(self)

        def start(self, task_group) -> None:
            del task_group
            self.started = True

        async def stop(self) -> None:
            self.stopped = True

        async def handle_command(self, envelope):
            del envelope
            return None

        async def reset_device(self, device_id):
            del device_id

    monkeypatch.setattr(factory_module, "ElgatoDeviceSupervisor", FakeSupervisor)
    plan = resolve_component_host_plan(
        _component_document(),
        definitions={"dev.deckr.hardware.elgato": factory_module.component},
    )

    async with mock_deckr(
        lane_contracts=plan.lane_contracts,
        lanes=plan.lane_names,
    ) as deckr, start_components(deckr, plan) as component_host:
        await _wait_for_readiness(
            component_host.component_manager,
            "dev.deckr.hardware.elgato:main",
        )
        supervisor = instances[0]
        assert supervisor.started
        assert supervisor.manager_id == "elgato-main"
        assert supervisor.runtime.endpoint.address == hardware_manager_address(
            "elgato-main"
        )
        assert supervisor.runtime.endpoint.metadata["endpointSlot"] == (
            "hardware_manager"
        )
        payload = await _wait_for_hardware_payload(deckr)
        assert payload.labels == {"room": "office"}

    assert instances[0].stopped


async def test_device_descriptor_covers_keys_touch_buttons_dials_and_touchscreen():
    raw = _FakeStreamDeck(
        serial="PLUS123",
        path="usb-plus",
        deck_type="Stream Deck +",
        rows=2,
        cols=4,
        key_count=8,
        touch_key_count=2,
        dial_count=4,
        touch=True,
        key_size=(120, 120),
        touchscreen_size=(800, 100),
    )
    device = ElgatoDockDevice(raw)
    descriptor = device.descriptor
    controls = {control.control_id: control for control in descriptor.controls}

    assert descriptor.device_id == "PLUS123"
    assert controls["key.0.0"].kind == "bitmap_key"
    assert controls["button.0"].kind == "button"
    assert controls["dial.0"].kind == "rotary_encoder"
    assert controls["touchscreen.0"].kind == "touch_surface"
    assert controls["key.0.0"].output_capabilities[0].constraints[0].value == 120
    assert controls["dial.0"].input_capabilities[0].capability_id == "encoder.relative"
    assert descriptor.default_status_indicator is not None


async def test_device_executes_raster_and_power_commands():
    raw = _FakeStreamDeck(serial="MINI123", path="usb-mini")
    device = ElgatoDockDevice(raw)
    raw.open()

    set_frame = hw_messages.control_command_message(
        controller_id="controller-main",
        sender_session_id="controller-session",
        manager_id="manager-main",
        device_id=device.id,
        control_id="key.0.0",
        capability_id="raster.bitmap",
        command_type="set_frame",
        params={"image": _png_payload(), "encoding": "png"},
    )
    assert await device.handle_command(set_frame, manager_id="manager-main")
    assert raw.key_images[0]

    clear = hw_messages.control_command_message(
        controller_id="controller-main",
        sender_session_id="controller-session",
        manager_id="manager-main",
        device_id=device.id,
        control_id="key.0.0",
        capability_id="raster.bitmap",
        command_type="clear",
    )
    assert await device.handle_command(clear, manager_id="manager-main")
    assert raw.key_images[0] is None

    sleep = hw_messages.control_command_message(
        controller_id="controller-main",
        sender_session_id="controller-session",
        manager_id="manager-main",
        device_id=device.id,
        capability_id="device.power",
        command_type="sleep",
    )
    wake = hw_messages.control_command_message(
        controller_id="controller-main",
        sender_session_id="controller-session",
        manager_id="manager-main",
        device_id=device.id,
        capability_id="device.power",
        command_type="wake",
    )
    assert await device.handle_command(sleep, manager_id="manager-main")
    assert await device.handle_command(wake, manager_id="manager-main")
    assert raw.brightness_values == [0, 100]


async def test_supervisor_advertises_all_devices_and_routes_claimed_input():
    raw_a = _FakeStreamDeck(serial="A123", path="usb-a")
    raw_b = _FakeStreamDeck(serial="B123", path="usb-b")
    async with mock_deckr() as deckr:
        manager_cm = deckr.endpoint(hardware_manager_address("manager-main"))
        controller_cm = deckr.endpoint(controller_address("controller-main"))
        manager_endpoint = await manager_cm.__aenter__()
        controller_endpoint = await controller_cm.__aenter__()
        try:
            runtime = HardwareManagerRuntime(
                endpoint=manager_endpoint,
                beacon=deckr.beacon,
                concord=deckr.concord,
                manager_id="manager-main",
            )
            supervisor = ElgatoDeviceSupervisor(
                runtime=runtime,
                manager_id="manager-main",
                device_manager_factory=lambda: _FakeDeviceManager([raw_a, raw_b]),
                poll_interval=0.01,
            )
            async with anyio.create_task_group() as tg:
                supervisor.start(tg)
                with anyio.fail_after(2):
                    while len(supervisor.devices) != 2:
                        await anyio.sleep(0.01)

                payload = await _wait_for_hardware_payload(
                    deckr,
                    device_ids={"A123", "B123"},
                )
                assert sorted(payload.devices) == ["A123", "B123"]

                contract = await _claim(
                    runtime,
                    deckr,
                    controller_endpoint,
                    device_id="A123",
                )
                assert (await deckr.concord._validate(contract)).status == (
                    ContractValidityStatus.VALID
                )
                deckr._message_bus.publish.reset_mock()  # noqa: SLF001

                active = supervisor.devices["A123"]
                await active._on_key_event(raw_a, 0, True)  # noqa: SLF001
                with anyio.fail_after(2):
                    while not deckr._message_bus.publish.called:  # noqa: SLF001
                        await anyio.sleep(0.01)
                routed = deckr._message_bus.publish.call_args.args[0]  # noqa: SLF001
                assert routed.recipient.endpoint == controller_endpoint.address
                assert routed.recipient_session_id == controller_endpoint.session_id

                await supervisor.stop()
                tg.cancel_scope.cancel()
        finally:
            await manager_cm.__aexit__(None, None, None)
            await controller_cm.__aexit__(None, None, None)
