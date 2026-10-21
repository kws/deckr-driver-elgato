from __future__ import annotations

import base64
import io
from importlib.metadata import entry_points
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
    HardwareCommandContext,
    HardwareManagerRuntime,
)
from message_bus_mocks import mock_deckr
from PIL import Image
from StreamDeck.Devices.StreamDeck import DialEventType, TouchscreenEventType

from deckr.drivers.elgato import _factory as factory_module
from deckr.drivers.elgato._device import ElgatoDockDevice
from deckr.drivers.elgato._discovery import ElgatoDeviceSupervisor

pytestmark = pytest.mark.asyncio


async def test_delayed_frame_does_not_restore_output_after_claim_cancellation(
    monkeypatch,
):
    raw = _FakeStreamDeck(serial="A123", path="usb-a")
    raw.open()
    device = ElgatoDockDevice(raw)
    async with (
        mock_deckr() as deckr,
        deckr.endpoint(hardware_manager_address("manager-main")) as manager,
        deckr.endpoint(controller_address("controller-main")) as controller,
        deckr.hardware_manager(manager, manager_id="manager-main") as runtime,
    ):
        runtime.command_handler = lambda context: device.handle_command(
            context, manager_id="manager-main"
        )

        async def reset(_device_id):
            await device.reset_outputs()

        runtime.reset_handler = reset
        await runtime.set_device(device.descriptor)
        contract = await _claim(runtime, deckr, controller)
        command = hw_messages.control_command_message(
            controller_id="controller-main",
            sender_session_id=controller.session_id,
            manager_id="manager-main",
            device_id=device.id,
            control_id="key.0.0",
            capability_id="raster.bitmap",
            command_type="set_frame",
            params={"image": _png_payload(), "encoding": "png"},
            recipient_session_id=runtime.endpoint.session_id,
            contract=contract.pointer,
        )
        paused, resume = anyio.Event(), anyio.Event()
        original = anyio.to_thread.run_sync

        async def delayed_thread(function, *args, **kwargs):
            if function.__name__ == "guarded":
                paused.set()
                await resume.wait()
            return await original(function, *args, **kwargs)

        monkeypatch.setattr(anyio.to_thread, "run_sync", delayed_thread)
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(runtime._handle_command, command)
            with anyio.fail_after(2):
                await paused.wait()
            await deckr.concord._cancel(contract, controller.address, reason="test")
            await deckr.concord.wait_current()
            await runtime._reconcile_claims(reason="test cancellation")
            assert raw.key_images[0] is None
            resume.set()
        assert raw.key_images[0] is None
    await device.mark_disconnected()
    raw.close()


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
        participant_sessions={
            str(runtime.endpoint.address): runtime.endpoint.session_id,
        },
    )
    await deckr.concord._attach(
        contract,
        controller_endpoint.address,
        controller_endpoint.session_id,
    )
    await runtime._claim_manager.notify_policy_changed()
    with anyio.fail_after(2):
        while (
            await deckr.concord._validate(contract)
        ).status != ContractValidityStatus.VALID or not any(
            claim.contract.pointer == contract.pointer for claim in runtime.live_claims
        ):
            await anyio.sleep(0)
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
        definitions={
            entry.name: entry.load()
            for entry in entry_points(
                group="deckr.components", name="dev.deckr.hardware.elgato"
            )
        },
    )

    async with (
        mock_deckr(
            lane_contracts=plan.lane_contracts,
            lanes=plan.lane_names,
        ) as deckr,
        start_components(deckr, plan) as component_host,
    ):
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
    assert await device.handle_command(
        HardwareCommandContext(set_frame, device.descriptor, lambda: None),
        manager_id="manager-main",
    )
    with Image.open(io.BytesIO(raw.key_images[0])) as rendered:
        assert rendered.format == "JPEG"
        assert rendered.size == (80, 80)
        red, green, blue = rendered.convert("RGB").getpixel((40, 40))
        assert red > 250 and green < 5 and blue < 5

    clear = hw_messages.control_command_message(
        controller_id="controller-main",
        sender_session_id="controller-session",
        manager_id="manager-main",
        device_id=device.id,
        control_id="key.0.0",
        capability_id="raster.bitmap",
        command_type="clear",
    )
    assert await device.handle_command(
        HardwareCommandContext(clear, device.descriptor, lambda: None),
        manager_id="manager-main",
    )
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
    assert await device.handle_command(
        HardwareCommandContext(sleep, device.descriptor, lambda: None),
        manager_id="manager-main",
    )
    assert await device.handle_command(
        HardwareCommandContext(wake, device.descriptor, lambda: None),
        manager_id="manager-main",
    )
    assert raw.brightness_values == [0, 100]


@pytest.mark.parametrize(
    ("manager_id", "device_id", "control_id", "image"),
    [
        ("other-manager", "MINI123", "key.0.0", _png_payload()),
        ("manager-main", "other-device", "key.0.0", _png_payload()),
        ("manager-main", "MINI123", "missing", _png_payload()),
        ("manager-main", "MINI123", "key.0.0", "invalid base64!"),
        ("manager-main", "MINI123", "key.0.0", "bm90IGFuIGltYWdl"),
    ],
)
async def test_invalid_or_misdirected_raster_commands_do_not_write_hardware(
    manager_id, device_id, control_id, image
):
    raw = _FakeStreamDeck(serial="MINI123", path="usb-mini")
    device = ElgatoDockDevice(raw)
    raw.open()
    command = hw_messages.control_command_message(
        controller_id="controller-main",
        sender_session_id="controller-session",
        manager_id=manager_id,
        device_id=device_id,
        control_id=control_id,
        capability_id="raster.bitmap",
        command_type="set_frame",
        params={"image": image, "encoding": "png"},
    )

    assert (
        await device.handle_command(
            HardwareCommandContext(command, device.descriptor, lambda: None),
            manager_id="manager-main",
        )
        is None
    )
    assert raw.key_images == {}
    assert raw.brightness_values == []


async def test_dial_and_touch_callbacks_deliver_values_and_ignore_invalid_input():
    raw = _FakeStreamDeck(
        serial="PLUS123",
        path="usb-plus",
        dial_count=1,
        touch=True,
        touchscreen_size=(800, 100),
    )
    device = ElgatoDockDevice(raw)
    # Build the descriptor as discovery does, including the touchscreen surface.
    _ = device.descriptor
    await device._on_dial_event(raw, 0, DialEventType.TURN, -2)
    await device._on_dial_event(raw, 0, DialEventType.TURN, 3)
    await device._on_dial_event(raw, 0, DialEventType.PUSH, True)
    await device._on_dial_event(raw, 0, DialEventType.PUSH, False)
    await device._on_dial_event(raw, 0, DialEventType.TURN, 0)
    await device._on_dial_event(raw, 99, DialEventType.TURN, 1)
    await device._on_touchscreen_event(raw, TouchscreenEventType.SHORT, {"x": 10})
    await device._on_touchscreen_event(
        raw, TouchscreenEventType.DRAG, {"x": 70, "x_out": 20}
    )
    await device._on_touchscreen_event(
        raw, TouchscreenEventType.DRAG, {"x": 20, "x_out": 70}
    )
    await device._on_touchscreen_event(
        raw, TouchscreenEventType.DRAG, {"x": 20, "x_out": 20}
    )
    await device.mark_disconnected()
    events = [event async for event in device.subscribe()]

    assert [
        (event.control_id, event.capability_id, event.event_type, event.value)
        for event in events
    ] == [
        (
            "dial.0",
            "encoder.relative",
            "rotate",
            {"delta": -2, "direction": "counterclockwise"},
        ),
        (
            "dial.0",
            "encoder.relative",
            "rotate",
            {"delta": 3, "direction": "clockwise"},
        ),
        ("dial.0", "button.momentary", "down", {"eventType": "down"}),
        ("dial.0", "button.momentary", "up", {"eventType": "up"}),
        ("touchscreen.0", "touch.gesture", "tap", {"eventType": "tap"}),
        (
            "touchscreen.0",
            "touch.gesture",
            "swipe",
            {"eventType": "swipe", "direction": "left"},
        ),
        (
            "touchscreen.0",
            "touch.gesture",
            "swipe",
            {"eventType": "swipe", "direction": "right"},
        ),
    ]


async def test_duplicate_interface_cleanup_preserves_active_device_and_claim():
    primary = _FakeStreamDeck(serial="A123", path="usb-primary")
    duplicate = _FakeStreamDeck(serial="A123", path="usb-duplicate")
    primary.open()
    device = ElgatoDockDevice(primary)
    async with (
        mock_deckr() as deckr,
        deckr.endpoint(hardware_manager_address("manager-main")) as manager,
        deckr.endpoint(controller_address("controller-main")) as controller,
        deckr.hardware_manager(manager, manager_id="manager-main") as runtime,
    ):
        supervisor = ElgatoDeviceSupervisor(runtime=runtime, manager_id="manager-main")
        assert await supervisor._register_device("primary", device)
        await runtime.set_device(device.descriptor)
        contract = await _claim(runtime, deckr, controller)

        with anyio.fail_after(2):
            await supervisor._device_loop(duplicate, "duplicate")

        assert duplicate.closed
        assert duplicate.reset_count == 0
        assert supervisor.devices == {"A123": device}
        assert primary.is_open()
        assert (await deckr.concord._validate(contract)).status == (
            ContractValidityStatus.VALID
        )
        payload = await _wait_for_hardware_payload(deckr, device_ids={"A123"})
        assert payload.devices["A123"].descriptor == device.descriptor

        # The owning interface must still remove the device and its authority.
        await supervisor._unregister_device("primary")
        assert supervisor.devices == {}
        assert (await deckr.concord._validate(contract)).status == (
            ContractValidityStatus.CANCELLED
        )
    await device.mark_disconnected()
    primary.close()


async def test_supervisor_advertises_all_devices_and_routes_claimed_input():
    raw_a = _FakeStreamDeck(serial="A123", path="usb-a")
    raw_b = _FakeStreamDeck(serial="B123", path="usb-b")
    async with mock_deckr() as deckr:
        manager_cm = deckr.endpoint(hardware_manager_address("manager-main"))
        controller_cm = deckr.endpoint(controller_address("controller-main"))
        manager_endpoint = await manager_cm.__aenter__()
        controller_endpoint = await controller_cm.__aenter__()
        try:
            runtime_context = deckr.hardware_manager(
                manager_endpoint,
                manager_id="manager-main",
                command_handler=lambda context: supervisor.handle_command(context),
                reset_handler=lambda device_id: supervisor.reset_device(device_id),
            )
            runtime = await runtime_context.__aenter__()
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

                unauthorized = hw_messages.control_command_message(
                    controller_id="controller-main",
                    sender_session_id=controller_endpoint.session_id,
                    manager_id="manager-main",
                    device_id="A123",
                    control_id="key.0.0",
                    capability_id="raster.bitmap",
                    command_type="clear",
                    recipient_session_id=runtime.endpoint.session_id,
                    contract={"contractId": "unauthorized-claim", "generation": 1},
                )
                deckr._message_bus.publish_reply.reset_mock()  # noqa: SLF001
                assert not await runtime._handle_command(unauthorized)  # noqa: SLF001
                rejected = deckr._message_bus.publish_reply.call_args.args[0]  # noqa: SLF001
                rejection = hw_messages.hardware_body_from_message(rejected)
                assert isinstance(rejection, hw_messages.CommandRejectedMessage)
                assert rejection.reason == "unauthorized"

                contract = await _claim(
                    runtime,
                    deckr,
                    controller_endpoint,
                    device_id="A123",
                )
                assert (await deckr.concord._validate(contract)).status == (
                    ContractValidityStatus.VALID
                )
                command = hw_messages.control_command_message(
                    controller_id="controller-main",
                    sender_session_id=controller_endpoint.session_id,
                    manager_id="manager-main",
                    device_id="A123",
                    control_id="key.0.0",
                    capability_id="raster.bitmap",
                    command_type="set_frame",
                    params={"image": _png_payload(), "encoding": "png"},
                    recipient_session_id=runtime.endpoint.session_id,
                    contract=contract.pointer,
                )
                assert await runtime._handle_command(command)
                with Image.open(io.BytesIO(raw_a.key_images[0])) as rendered:
                    assert rendered.size == (80, 80)
                    assert rendered.convert("RGB").getpixel((40, 40))[0] > 250
                assert raw_b.key_images == {}
                deckr._message_bus.publish.reset_mock()  # noqa: SLF001

                active = supervisor.devices["A123"]
                await active._on_key_event(raw_a, 0, True)  # noqa: SLF001
                with anyio.fail_after(2):
                    while not deckr._message_bus.publish.called:  # noqa: SLF001
                        await anyio.sleep(0.01)
                routed = deckr._message_bus.publish.call_args.args[0]  # noqa: SLF001
                assert routed.recipient.endpoint == controller_endpoint.address
                assert routed.recipient_session_id == controller_endpoint.session_id
                body = hw_messages.hardware_body_from_message(routed)
                assert isinstance(body, hw_messages.ControlInputMessage)
                assert body.control_id == "key.0.0"
                assert body.capability_id == "button.momentary"
                assert body.event_type == "down"

                raw_a.close()
                with anyio.fail_after(2):
                    while (await deckr.concord._validate(contract)).status != (  # noqa: SLF001
                        ContractValidityStatus.CANCELLED
                    ):
                        await anyio.sleep(0.01)
                await runtime.wait_for_state(
                    lambda state: not state.claims, timeout_seconds=2
                )
                assert runtime.live_claims == ()
                assert raw_a.reset_count == 1

                await supervisor.stop()
                tg.cancel_scope.cancel()
        finally:
            if "runtime_context" in locals():
                await runtime_context.__aexit__(None, None, None)
            await manager_cm.__aexit__(None, None, None)
            await controller_cm.__aexit__(None, None, None)
