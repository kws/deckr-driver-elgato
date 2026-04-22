# deckr-driver-elgato

Elgato Stream Deck driver package for Deckr.

This repo publishes the `deckr-driver-elgato` distribution while keeping the runtime
module surface under `deckr.drivers.elgato`.

## Development

Build a local `deckr` wheel first:

```bash
cd ../deckr && uv build --wheel
cd ../deckr-driver-elgato
uv sync --dev --find-links ../deckr/dist
uv run --find-links ../deckr/dist pytest
```
