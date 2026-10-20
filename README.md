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

The `deckr` Python distribution now comes from the sibling
[`deckr-core-python`](https://github.com/kws/deckr-core-python) repository.
Check it out next to this repository at the revision in `.deckr-core-revision`. In the parent workspace, run
`uv sync --all-packages --all-groups`, then the repository-local checks:

```bash
uv run --no-sync ruff check .
uv run --no-sync pytest
```

For standalone development, build a wheel from the separated core and install
it without the parent workspace source override:

```bash
(cd ../deckr-core-python && uv build --wheel)
uv sync --no-sources --dev --find-links ../deckr-core-python/dist
uv run --no-sync pytest
```

Runtime imports remain under `deckr`; the reference-only `deckr` repository is
not a Python package dependency.
