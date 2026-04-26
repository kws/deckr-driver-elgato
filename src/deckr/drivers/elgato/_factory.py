import anyio
from deckr.contracts.messages import hardware_manager_address
from deckr.core.component import BaseComponent, RunContext
from deckr.core.components import (
    ComponentContext,
    ComponentDefinition,
    ComponentManifest,
)
from deckr.transports.bus import EventBus

from deckr.drivers.elgato._discovery import discover_elgato_devices


class ElgatoDeviceFactory(BaseComponent):
    def __init__(self, event_bus: EventBus, *, manager_id: str):
        super().__init__("elgato_device_factory")
        self.event_bus = event_bus
        self.manager_id = manager_id
        self.__cancel_scope = None

    async def start(self, ctx: RunContext) -> None:
        endpoint = str(hardware_manager_address(self.manager_id))
        client_id = await self.event_bus.claim_local_endpoint(endpoint)
        async with anyio.create_task_group() as tg:
            try:
                self.__cancel_scope = tg.cancel_scope
                async with discover_elgato_devices(
                    self.event_bus,
                    manager_id=self.manager_id,
                ) as stream:
                    async for event in stream:
                        await self.event_bus.send(event)
            finally:
                await self.event_bus.withdraw_local_endpoint(
                    endpoint=endpoint,
                    client_id=client_id,
                )

    async def stop(self) -> None:
        with anyio.CancelScope(shield=True):
            if self.__cancel_scope is not None:
                self.__cancel_scope.cancel()


def driver_factory(event_bus: EventBus, *, manager_id: str) -> ElgatoDeviceFactory:
    return ElgatoDeviceFactory(event_bus=event_bus, manager_id=manager_id)


def component_factory(context: ComponentContext) -> ElgatoDeviceFactory:
    source = dict(context.raw_config)
    manager_id = str(source.get("manager_id", "")).strip()
    if not manager_id:
        raise ValueError("deckr.drivers.elgato requires manager_id")
    return driver_factory(
        context.require_lane("hardware_events"),
        manager_id=manager_id,
    )


component = ComponentDefinition(
    manifest=ComponentManifest(
        component_id="deckr.drivers.elgato",
        config_prefix="deckr.drivers.elgato",
        consumes=("hardware_events",),
        publishes=("hardware_events",),
    ),
    factory=component_factory,
)
