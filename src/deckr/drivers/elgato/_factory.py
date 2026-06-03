"""Deckr component entry point for the Elgato hardware manager."""

from __future__ import annotations

import logging
from collections.abc import Mapping

from deckr.components import (
    BaseComponent,
    ComponentContext,
    ComponentDefinition,
    ComponentManifest,
    RunContext,
)
from deckr.hardware.runtime import HardwareManagerRuntime

from deckr.drivers.elgato._discovery import ElgatoDeviceSupervisor

logger = logging.getLogger(__name__)


def _labels_from_config(config: Mapping[str, object] | None) -> dict[str, str]:
    raw = dict(config or {}).get("labels", {})
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("Elgato manager config.labels must be a table")
    labels: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Elgato manager config.labels keys must be strings")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Elgato manager config.labels.{key} must be a non-empty string"
            )
        labels[key.strip()] = value.strip()
    return labels


class ElgatoHardwareComponent(BaseComponent):
    def __init__(self, context: ComponentContext) -> None:
        super().__init__(context.runtime_name)
        self._context = context
        self._runtime: HardwareManagerRuntime | None = None
        self._supervisor: ElgatoDeviceSupervisor | None = None

    async def start(self, ctx: RunContext) -> None:
        ctx.start_task(self._run, ctx)

    async def stop(self) -> None:
        supervisor = self._supervisor
        if supervisor is not None:
            await supervisor.stop()
        runtime = self._runtime
        if runtime is not None:
            await runtime.stop()

    async def _run(self, ctx: RunContext) -> None:
        async with self._context.open_endpoint(
            "hardware_manager",
            metadata={"runtime": "deckr-driver-elgato-python"},
        ) as endpoint:
            runtime = HardwareManagerRuntime(
                endpoint=endpoint,
                beacon=self._context.require_beacon(),
                concord=self._context.require_concord(),
                manager_id=endpoint.address.endpoint_id,
                labels=_labels_from_config(self._context.config),
            )
            supervisor = ElgatoDeviceSupervisor(
                runtime=runtime,
                manager_id=endpoint.address.endpoint_id,
            )
            runtime.command_handler = supervisor.handle_command
            runtime.reset_handler = supervisor.reset_device
            self._runtime = runtime
            self._supervisor = supervisor
            try:
                await runtime.start(ctx.tg)
                supervisor.start(ctx.tg)
                await ctx.report_ready()
                await ctx.stopping.wait()
            finally:
                await supervisor.stop()
                await runtime.stop()
                self._runtime = None
                self._supervisor = None


def component_factory(context: ComponentContext) -> ElgatoHardwareComponent:
    return ElgatoHardwareComponent(context)


component = ComponentDefinition(
    manifest=ComponentManifest(
        component_id="dev.deckr.hardware.elgato",
        consumes=("hardware_messages",),
        publishes=("hardware_messages",),
        endpoint_slots=("hardware_manager",),
        role="hardware_manager",
    ),
    factory=component_factory,
)
