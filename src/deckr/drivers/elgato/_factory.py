import anyio
from deckr.core.component import BaseComponent, RunContext
from deckr.core.components import (
    ComponentContext,
    ComponentDefinition,
    ComponentManifest,
)
from deckr.core.messaging import EventBus

from deckr.drivers.elgato._discovery import discover_elgato_devices


class ElgatoDeviceFactory(BaseComponent):
    def __init__(self, event_bus: EventBus):
        super().__init__("elgato_device_factory")
        self.event_bus = event_bus
        self.__cancel_scope = None

    async def start(self, ctx: RunContext) -> None:
        async with anyio.create_task_group() as tg:
            self.__cancel_scope = tg.cancel_scope
            async with discover_elgato_devices() as stream:
                async for event in stream:
                    await self.event_bus.send(event)

    async def stop(self) -> None:
        with anyio.CancelScope(shield=True):
            if self.__cancel_scope is not None:
                self.__cancel_scope.cancel()


def driver_factory(event_bus: EventBus) -> ElgatoDeviceFactory:
    return ElgatoDeviceFactory(event_bus=event_bus)


def component_factory(context: ComponentContext) -> ElgatoDeviceFactory:
    return driver_factory(context.require_lane("hardware_events"))


component = ComponentDefinition(
    manifest=ComponentManifest(
        component_id="deckr.drivers.elgato",
        config_prefix="deckr.drivers.elgato",
        publishes=("hardware_events",),
    ),
    factory=component_factory,
)
