# Frame — Motion Poster Studio

A self-hosted motion poster editor. Upload artwork, choose a motion style, add crisp text, and export an MP4. All video rendering runs locally on the CPU. **No paid video generation or external video API is supported.**

## Run

```bash
cp .env.example .env
docker compose up -d --build
```

Open **http://localhost:4173** on the instance. Forward port 4173 with SSH or your editor when accessing a remote instance. The Compose port is bound to loopback by default. Frontend and backend share this port.

## Architecture learning guide

Open **http://localhost:4173/architecture.html**, or open [frontend/architecture.html](frontend/architecture.html) directly in a browser. The single-file guide works offline with no external fonts, scripts, images, or model requests. It includes:

- A clickable SVG diagram comparing the implemented app with the proposed AI architecture.
- Six AI engineering patterns with explanations and primary documentation.
- Step-by-step simulations for normal work, quota exhaustion, invalid plans, and restart recovery.
- Proposed ScenePlan contract, evaluation records, and phased implementation.
- Review of the user's Wix video-template references and original schematic design interpretations.
- HTML and SVG downloads, plus a print/PDF layout.

The proposed architecture is **AI-assisted planning → validated scene data → local CPU rendering**. Typed planners, LangGraph checkpoints, richer templates, worker leases, atomic publication, and automated quality checks are future work. The guide labels implemented and proposed behavior explicitly.

## What local rendering does

FFmpeg animates existing artwork using slow zoom, horizontal pan, or soft float, then composites the text overlay and encodes H.264 MP4. It does not synthesize new scenes or objects. Exports are silent. No GPU, hosted video model, or API key is required. There are no video API charges; your instance still uses compute and storage.

The frontend uses Canvas for preview and PNG overlay preparation. FastAPI validates requests. SQLite stores job snapshots and prompt cache results. A bounded in-process queue renders one job at a time. Startup requeues unfinished local jobs; this is not a persistent graph workflow. Use one Uvicorn worker.

A persistent Docker volume stores uploaded images, overlays, jobs, and completed videos under `/app/data`. Requests with the same content fingerprint reuse a queued or completed job. Unknown/external video modes are rejected, including when an xAI key exists in the environment. Legacy unfinished external-video jobs are stopped without a provider call.

## Optional NVIDIA creative assistance

`NVIDIA_API_KEY` enables the explicit “Refine with NVIDIA” button. `NVIDIA_TEXT_MODEL` selects the hosted text model. The endpoint sends a written artwork description, not the image, to produce creative direction notes. **The current renderer follows the selected preset; it does not interpret those notes as a scene plan.**

Hosted text assistance has separate account limits and potentially billing. It is never required for local video rendering. Groq and xAI/Grok are different services. No Groq integration is currently implemented; the architecture guide proposes it only for optional text planning with verified account access. There is no automatic paid fallback.

Store keys only in the ignored `.env` or Dokploy runtime environment. Restart after changes:

```bash
docker compose up -d --force-recreate
```

## Access and Dokploy

This is a **single-owner app**, not a multi-tenant service. Set `APP_PASSWORD` before exposing it beyond localhost; HTTP Basic username is `studio`. All authenticated visitors share one clip library. Use HTTPS for network access.

In Dokploy, create an Application from this GitHub repository using its Dockerfile, container port **4173**, and a persistent volume at **`/app/data`**. Configure `APP_PASSWORD` and optional NVIDIA credentials in runtime settings. Add a domain and HTTPS in Dokploy when needed. `compose.yaml` provides the local persistent service and configured limits of 1.5 CPU / 768 MB; measure actual capacity before increasing concurrency.

## Development and verification

Python 3.12 and FFmpeg are required outside Docker:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 4173
```

Run integration tests in an isolated temporary database, with hosted inference mocked:

```bash
docker compose run --rm -v "$PWD/tests:/app/tests:ro" studio python -m unittest discover -s tests -v
```

Tests verify real local MP4 rendering and byte-range downloads, request caching, invalid media, rejection of external video generation even with a key, a local render without network access, password protection, cross-origin write rejection, and optional prompt caching. Browser verification covers desktop/mobile editing, export, library, and the architecture guide. No paid video request is used.

## Layout

```text
frontend/architecture.html  Offline interactive learning guide
frontend/                   Editor, canvas preview, clip library
backend/app.py              API, SQLite jobs, CPU FFmpeg worker
backend/providers.py        Optional NVIDIA text assistance
Dockerfile / compose.yaml   Small-instance deployment
```
