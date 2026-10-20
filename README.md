# Frame — Motion Poster Studio

A self-hosted motion poster editor. Upload artwork, choose a motion style, add crisp text overlays, and export an MP4. Designed for a small CPU instance: AI inference stays with hosted providers.

## Run locally

```bash
cp .env.example .env
docker compose up -d --build
```

Open **http://localhost:4173** on the instance. For a remote instance, forward port 4173 with SSH or your editor. The Compose port binds to loopback by default.

Both frontend and backend are served from port 4173. No JavaScript build toolchain is required. FastAPI handles jobs and FFmpeg exports H.264 MP4. A persistent Docker volume stores uploaded images, overlays, job history, and videos. One worker renders at a time to limit CPU use. Use one Uvicorn worker.

## Modes and API keys

- **Local motion:** free from API charges. Slow zoom, horizontal pan, and soft float are deterministic animation, without generative scene changes. Exports are silent.
- **Grok Imagine:** image-to-video through `grok-imagine-video-1.5-lite`, preserving generated audio when supplied. Set `XAI_API_KEY` to an **xAI** key. A **Groq** key (`gsk_…`) is a different provider and cannot generate Grok videos.
- **NVIDIA creative assist:** generates a motion direction from your written artwork description. Set `NVIDIA_API_KEY` and optionally `NVIDIA_TEXT_MODEL`. The artwork itself is not sent for prompt refinement.

Keys belong in the ignored `.env` or Dokploy runtime environment. Never commit them. After editing `.env`, use `docker compose up -d --force-recreate`.

No free xAI API quota is assumed or promised. Any promotional credits depend on your account. Grok estimates use published Lite pricing: $0.02/sec at 480p, $0.03/sec at 720p, plus $0.01 for one input image. Verify current prices and model availability in your xAI account. The UI requires confirmation before a paid generation. `MAX_DAILY_USD` defaults to $2; reservations conservatively include failed requests because provider acceptance/billing can be ambiguous. It is an application estimate cap, not a provider billing guarantee. NVIDIA prompt calls have separate account limits.

## Reliability and privacy

Identical clip requests reuse the existing queued/completed job. Jobs survive restarts. A saved xAI request ID is polled again after restart; an interrupted submission without a request ID is never automatically repeated. Polling errors do not trigger a new billable generation. Provider response metadata stays server-side, while keys are never returned by the API. Stored media and SQLite are under `data/` (the Docker volume in production).

Uploads accept PNG/JPEG/WebP up to 6 MB / 16 megapixels. The browser prepares artwork and transparent text overlays at output resolution. Text is composited with FFmpeg after motion generation, keeping overlays stable and exact. Text already baked into an uploaded poster may still move or change with generative animation.

The studio is a **single-owner application**, not a multi-tenant service. Use `APP_PASSWORD` before exposing it beyond localhost; the HTTP Basic username is `studio`. Place it behind HTTPS in Dokploy. All visitors with that password share the same clip library and API budget.

## Dokploy

Create an Application from this GitHub repo with Dockerfile build, container port **4173**, and a persistent volume at **`/app/data`**. Set `APP_PASSWORD` and provider keys in Dokploy's environment settings. Add your domain and HTTPS there. Alternatively import `compose.yaml`; leave the host loopback binding in place for local use, and use Dokploy's internal service routing for a domain. This repo includes deployment configuration; creating a Dokploy domain is a separate step.

## Development and tests

Python 3.12 and FFmpeg are required outside Docker:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 4173
```

Run integration tests against an isolated temporary database and mocked paid providers:

```bash
docker compose run --rm -v "$PWD/tests:/app/tests:ro" studio python -m unittest discover -s tests -v
```

Tests cover MP4 encoding and range downloads, duplicate generation caching, invalid media, API key and cost confirmation, daily budget limits, paid submission retry prevention, password protection, cross-origin write rejection, and prompt caching. No paid video request is used for testing.

Initial instance verification: all eight integration tests passed; Chromium checked desktop and 390px mobile layouts, all three aspect ratios, text controls, clip history, and an actual MP4 download without browser errors. A live NVIDIA prompt request succeeded. Grok video generation remains unverified until a valid xAI key is configured.

## Project layout

```text
frontend/           Browser editor, canvas preview, clip library
backend/app.py      API, SQLite jobs, single render worker, FFmpeg
backend/providers.py Hosted inference interfaces
tests/              Integration and provider-boundary tests
Dockerfile          CPU-only runtime
compose.yaml        Local persistent service / Dokploy starting point
```
