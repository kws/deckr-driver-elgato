# deckr-driver-elgato

Elgato Stream Deck hardware manager package for Deckr.

This repo publishes the `deckr-driver-elgato` distribution while keeping the runtime
module surface under `deckr.drivers.elgato`.

## Runtime

The component is hosted by Deckr through the `hardware_manager` endpoint slot,
using `ComponentContext.open_endpoint()`, `require_beacon()`, and
`require_concord()`. Device discovery and USB command execution are local to
this package, while Beacon candidate advertisement, Concord hardware-claim
participation, token refresh, command authorization, and input routing are
handled by `deckr.hardware.runtime.HardwareManagerRuntime`.

The manager no longer publishes legacy discovery inventory or lease-state claim
records. Controllers claim devices through `dev.deckr.profile.hardware_claim.v1`
Concord contracts, and the manager routes input only while the matching Concord
claim is valid.

The supervisor opens every attached Elgato Stream Deck, advertises a Deckr
descriptor for its keys, touch buttons, dials, touch surfaces, and display
surfaces, then forwards claimed hardware input through the runtime. Raster
bitmap commands are converted with `StreamDeck.ImageHelpers.PILHelper`, so
Pillow is a direct runtime dependency.

## Development

Build a local `deckr` wheel first:

```bash
cd ../deckr && uv build --wheel
cd ../deckr-driver-elgato
uv sync --dev --find-links ../deckr/dist
uv run --find-links ../deckr/dist pytest
```
