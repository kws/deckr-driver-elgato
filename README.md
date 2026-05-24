# deckr-driver-elgato

Elgato Stream Deck hardware manager package for Deckr.

This repo publishes the `deckr-driver-elgato` distribution while keeping the runtime
module surface under `deckr.drivers.elgato`.

## Runtime

The component registers `hardware_manager:<manager-id>` on the
`hardware_messages` lane. Device discovery and USB command execution are local
to this package, while Beacon advertisement, Concord hardware-claim
participation, token refresh, and command/input routing are handled by
`deckr.hardware.runtime.HardwareManagerRuntime`.

The manager no longer publishes legacy discovery inventory or lease-state claim
records. Controllers claim devices through `dev.deckr.profile.hardware_claim.v1`
Concord contracts, and the manager routes input only while the matching Concord
claim is valid.

## Development

Build a local `deckr` wheel first:

```bash
cd ../deckr && uv build --wheel
cd ../deckr-driver-elgato
uv sync --dev --find-links ../deckr/dist
uv run --find-links ../deckr/dist pytest
```
