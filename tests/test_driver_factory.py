from __future__ import annotations

import anyio
import pytest
from deckr.core.messaging import EventBus

from deckr.drivers.elgato._discovery import discover_loop
from deckr.drivers.elgato._factory import ElgatoDeviceFactory, driver_factory


class _FakeDeviceManager:
    def __init__(self, devices: list[object]) -> None:
        self._devices = devices

    def enumerate(self) -> list[object]:
        return self._devices


def test_driver_factory_returns_elgato_device_factory() -> None:
    factory = driver_factory(EventBus())

    assert isinstance(factory, ElgatoDeviceFactory)


@pytest.mark.asyncio
async def test_discover_loop_emits_first_available_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_device = object()
    monkeypatch.setattr(
        "deckr.drivers.elgato._discovery.DeviceManager",
        lambda: _FakeDeviceManager([fake_device]),
    )
    send_stream, receive_stream = anyio.create_memory_object_stream[object](
        max_buffer_size=1
    )
    device_connected = [False]

    async with anyio.create_task_group() as tg:
        tg.start_soon(discover_loop, send_stream, device_connected)

        with anyio.fail_after(2):
            assert await receive_stream.receive() is fake_device

        device_connected[0] = True
        with anyio.move_on_after(0.2) as scope:
            await receive_stream.receive()
        assert scope.cancel_called is True

        tg.cancel_scope.cancel()
