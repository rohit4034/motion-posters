import asyncio
import base64
import hashlib
import hmac
import io
import json
import os
import secrets
import shutil
import sqlite3
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from .providers import NvidiaTextProvider, ProviderError, XaiVideoProvider

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
DATA = Path(os.getenv("DATA_DIR", str(ROOT / "data")))
DATA.mkdir(parents=True, exist_ok=True)
Image.MAX_IMAGE_PIXELS = 16_000_000
video_provider = XaiVideoProvider()
queue = asyncio.Queue(maxsize=16)


def db():
    connection = sqlite3.connect(DATA / "studio.db", timeout=15)
    connection.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, fingerprint TEXT, created REAL, payload TEXT)")
    connection.execute("CREATE TABLE IF NOT EXISTS prompts (fingerprint TEXT PRIMARY KEY, payload TEXT)")
    return connection


def save(job):
    with db() as c:
        c.execute("INSERT OR REPLACE INTO jobs VALUES (?,?,?,?)", (job["id"], job["fingerprint"], job["created"], json.dumps(job)))


def get_job(job_id):
    with db() as c:
        row = c.execute("SELECT payload FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Clip not found")
    return json.loads(row[0])


def list_jobs():
    with db() as c:
        return [json.loads(r[0]) for r in c.execute("SELECT payload FROM jobs ORDER BY created DESC")]


def public_job(job):
    return {k: v for k, v in job.items() if k not in ("fingerprint", "request_id", "upstream", "spec")}


def estimated_cost(duration, resolution):
    return round(duration * {"480p": .02, "720p": .03}[resolution] + .01, 2)


def spent_today():
    today = datetime.now(timezone.utc).date()
    return round(sum(j.get("estimated_usd", 0) for j in list_jobs()
                     if j["mode"] == "grok" and datetime.fromtimestamp(j["created"], timezone.utc).date() == today), 2)


def decode_image(value, path):
    if not value.startswith(("data:image/png;base64,", "data:image/jpeg;base64,", "data:image/webp;base64,")):
        raise HTTPException(422, "Use a PNG, JPEG, or WebP image.")
    try:
        raw = base64.b64decode(value.split(",", 1)[1], validate=True)
        if len(raw) > 6_000_000:
            raise HTTPException(413, "Image is too large. Maximum size is 6 MB.")
        with Image.open(io.BytesIO(raw)) as im:
            if im.width < 32 or im.height < 32 or im.width * im.height > 16_000_000:
                raise HTTPException(422, "Image dimensions must be at least 32 pixels, up to 16 megapixels.")
            im.convert("RGBA").save(path, "PNG")
    except (ValueError, UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(422, "This image could not be read. Try another PNG or JPEG.")


class ClipSpec(BaseModel):
    title: str = Field(default="Untitled motion", max_length=100)
    mode: Literal["local", "grok"] = "local"
    preset: Literal["zoom", "pan", "float"] = "zoom"
    duration: Literal[5, 10, 15] = 5
    aspect: Literal["9:16", "1:1", "16:9"] = "9:16"
    resolution: Literal["480p", "720p"] = "480p"
    prompt: str = Field(default="Slow cinematic push-in. Preserve the source artwork.", min_length=5, max_length=2000)
    image: str = Field(max_length=9_000_000)
    overlay: str = Field(max_length=9_000_000)
    confirm_cost: bool = False


class PromptSpec(BaseModel):
    brief: str = Field(min_length=5, max_length=1000)
    preset: Literal["zoom", "pan", "float"] = "zoom"


def dimensions(spec):
    edge = 480 if spec["resolution"] == "480p" else 720
    long = 854 if edge == 480 else 1280
    return {"9:16": (edge, long), "16:9": (long, edge), "1:1": (edge, edge)}[spec["aspect"]]


def render_clip(job, cloud=False):
    folder = DATA / job["id"]
    spec = job["spec"]
    w, h = dimensions(spec)
    frames = spec["duration"] * 24
    inputs = ["-i", str(folder / "source.mp4")] if cloud else ["-i", str(folder / "artwork.png")]
    if cloud:
        motion = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1,fps=24"
    else:
        z = {"zoom": f"1+0.09*on/{frames}", "pan": "1.10", "float": f"1.06+0.02*sin(on/{frames}*2*PI)"}[spec["preset"]]
        x = f"(iw-iw/zoom)*on/{frames}" if spec["preset"] == "pan" else "iw/2-iw/zoom/2"
        motion = (f"scale={w*2}:{h*2}:force_original_aspect_ratio=increase,crop={w*2}:{h*2},"
                  f"zoompan=z='{z}':x='{x}':y='ih/2-ih/zoom/2':d={frames}:s={w}x{h}:fps=24,setsar=1")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-threads", "1", *inputs,
               "-loop", "1", "-i", str(folder / "overlay.png"),
               "-filter_complex_threads", "1", "-filter_complex",
               f"[0:v]{motion}[bg];[1:v]scale={w}:{h},format=rgba[fg];[bg][fg]overlay=0:0:shortest=1,format=yuv420p[out]",
               "-map", "[out]"]
    if cloud:
        command += ["-map", "0:a?", "-c:a", "aac"]
    command += ["-t", str(spec["duration"]), "-c:v", "libx264", "-threads", "1", "-preset", "veryfast", "-crf", "22", "-movflags", "+faststart", str(folder / "clip.mp4")]
    subprocess.run(command, check=True, capture_output=True, timeout=240)


async def process(job):
    started = time.monotonic()
    folder = DATA / job["id"]
    try:
        job.update(status="rendering", progress=12, message="Preparing your artwork")
        save(job)
        if job["mode"] == "grok":
            if not job.get("request_id"):
                # Never automatically retry a submission: upstream acceptance can be ambiguous.
                job.update(status="submitting", message="Sending artwork to Grok", submission_attempted=True)
                save(job)
                image = "data:image/png;base64," + base64.b64encode((folder / "artwork.png").read_bytes()).decode()
                result = await video_provider.submit(job["spec"], image)
                job["request_id"] = result["request_id"]
                job["upstream"] = result
                save(job)
            job.update(status="generating", progress=30, message="Grok is animating your artwork. This may take a few minutes.")
            save(job)
            for index in range(180):
                await asyncio.sleep(5)
                result = await video_provider.status(job["request_id"])
                job["upstream"] = result
                if result.get("status") == "done":
                    url = result["video"]["url"]
                    parsed = urlparse(url)
                    if parsed.scheme != "https" or not (parsed.hostname or "").endswith(".x.ai"):
                        raise ProviderError("Unexpected provider download host. The upstream request is saved for recovery.")
                    async with httpx.AsyncClient(timeout=180, follow_redirects=False) as client:
                        async with client.stream("GET", url) as response:
                            response.raise_for_status()
                            total = 0
                            with (folder / "source.mp4").open("wb") as f:
                                async for chunk in response.aiter_bytes():
                                    total += len(chunk)
                                    if total > 100_000_000:
                                        raise ProviderError("Generated video exceeds the 100 MB download limit.")
                                    f.write(chunk)
                    break
                if result.get("status") in ("expired", "failed"):
                    raise ProviderError("Grok could not complete this generation. No paid retry was made.")
                job["progress"] = min(78, 30 + index)
                save(job)
            else:
                raise ProviderError("Generation timed out. Its request ID was saved; no paid retry was made.")
        job.update(status="rendering", progress=85, message="Rendering crisp text and preparing your MP4")
        save(job)
        await asyncio.to_thread(render_clip, job, job["mode"] == "grok")
        job.update(status="completed", progress=100, message="Your motion poster is ready", video_url=f"/api/clips/{job['id']}/video", latency_ms=round((time.monotonic()-started)*1000))
    except Exception as exc:
        message = str(exc) if isinstance(exc, ProviderError) else "The clip could not be completed. Check server logs; no automatic paid retry was made."
        print(f"Clip {job['id']} failed: {type(exc).__name__}", flush=True)
        job.update(status="failed", message=message, progress=0)
    finally:
        save(job)


async def worker():
    while True:
        job_id = await queue.get()
        try:
            await process(get_job(job_id))
        finally:
            queue.task_done()


@asynccontextmanager
async def lifespan(app):
    global queue
    queue = asyncio.Queue(maxsize=16)
    db().close()
    task = asyncio.create_task(worker())
    for job in reversed(list_jobs()):
        if job["status"] not in ("completed", "failed"):
            if job["mode"] == "grok" and job.get("submission_attempted") and not job.get("request_id"):
                job.update(status="failed", message="Server stopped during submission. Check xAI usage before generating again; submission was not retried.")
                save(job)
            else:
                await queue.put(job["id"])
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="Motion Poster Studio", lifespan=lifespan)


@app.middleware("http")
async def guard(request: Request, call_next):
    if request.url.path != "/health":
        password = os.getenv("APP_PASSWORD", "")
        if password:
            expected = "Basic " + base64.b64encode(f"studio:{password}".encode()).decode()
            if not hmac.compare_digest(request.headers.get("authorization", ""), expected):
                return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="Motion Poster Studio"'})
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin and urlparse(origin).netloc != request.headers.get("host"):
                return JSONResponse({"detail": "Cross-origin writes are not allowed"}, status_code=403)
            length = request.headers.get("content-length")
            if not length or not length.isdigit() or int(length) > 19_000_000:
                return JSONResponse({"detail": "Request must be smaller than 19 MB"}, status_code=413)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'"
    return response


@app.get("/health")
def health():
    return {"status": "ok", "renderer": bool(shutil.which("ffmpeg"))}


@app.get("/api/config")
def config():
    return {"grok_connected": bool(os.getenv("XAI_API_KEY")), "nvidia_connected": bool(os.getenv("NVIDIA_API_KEY")),
            "model": video_provider.model, "local_ready": bool(shutil.which("ffmpeg")),
            "daily_limit_usd": float(os.getenv("MAX_DAILY_USD", "2")), "reserved_today_usd": spent_today(),
            "pricing": {"480p": .02, "720p": .03, "image_input": .01}, "free_api_quota_verified": False}


@app.get("/api/clips")
def clips():
    return [public_job(j) for j in list_jobs()[:100]]


@app.get("/api/clips/{job_id}")
def clip(job_id: str):
    return public_job(get_job(job_id))


@app.get("/api/clips/{job_id}/artwork")
def artwork(job_id: str):
    job = get_job(job_id)
    return FileResponse(DATA / job["id"] / "artwork.png", media_type="image/png")


@app.get("/api/clips/{job_id}/video")
def video(job_id: str, download: bool = False):
    job = get_job(job_id)
    if job["status"] != "completed":
        raise HTTPException(409, "Clip is not ready")
    return FileResponse(DATA / job["id"] / "clip.mp4", media_type="video/mp4",
                        filename="motion-poster.mp4" if download else None)


@app.post("/api/clips", status_code=202)
async def create_clip(spec: ClipSpec):
    if not shutil.which("ffmpeg"):
        raise HTTPException(503, "MP4 renderer is not installed. Start the Docker Compose service.")
    if queue.full():
        raise HTTPException(429, "The render queue is full. Try again after a clip finishes.")
    if spec.mode == "grok" and not os.getenv("XAI_API_KEY"):
        raise HTTPException(503, "Grok is not connected. Add XAI_API_KEY to .env and restart. Local motion is free to use.")
    if spec.mode == "grok" and not spec.confirm_cost:
        raise HTTPException(422, "Confirm the estimated API cost before generating.")
    payload = spec.model_dump()
    fingerprint = hashlib.sha256(json.dumps({**payload, "model": video_provider.model}, sort_keys=True).encode()).hexdigest()
    for existing in list_jobs():
        if existing["fingerprint"] == fingerprint and existing["status"] != "failed":
            return public_job(existing)
    cost = estimated_cost(spec.duration, spec.resolution) if spec.mode == "grok" else 0
    if cost + spent_today() > float(os.getenv("MAX_DAILY_USD", "2")):
        raise HTTPException(429, "The daily Grok estimate limit has been reached. Local motion is still available.")
    job_id = uuid.uuid4().hex
    folder = DATA / job_id
    folder.mkdir()
    try:
        decode_image(spec.image, folder / "artwork.png")
        decode_image(spec.overlay, folder / "overlay.png")
    except Exception:
        shutil.rmtree(folder)
        raise
    payload.pop("image")
    payload.pop("overlay")
    job = {"id": job_id, "fingerprint": fingerprint, "created": time.time(), "title": spec.title,
           "mode": spec.mode, "preset": spec.preset, "duration": spec.duration, "aspect": spec.aspect,
           "resolution": spec.resolution, "estimated_usd": cost, "status": "queued", "progress": 0,
           "message": "Waiting for the renderer", "spec": payload}
    save(job)
    queue.put_nowait(job_id)
    return public_job(job)


@app.post("/api/prompt")
async def improve_prompt(spec: PromptSpec):
    fingerprint = hashlib.sha256(json.dumps(spec.model_dump(), sort_keys=True).encode()).hexdigest()
    with db() as c:
        row = c.execute("SELECT payload FROM prompts WHERE fingerprint=?", (fingerprint,)).fetchone()
    if row:
        return {**json.loads(row[0]), "cached": True}
    try:
        result = await NvidiaTextProvider().motion_prompt(spec.brief, spec.preset)
    except ProviderError as exc:
        raise HTTPException(502, str(exc))
    except httpx.HTTPError:
        raise HTTPException(502, "NVIDIA could not be reached. Your current motion direction is unchanged.")
    with db() as c:
        c.execute("INSERT OR REPLACE INTO prompts VALUES (?,?)", (fingerprint, json.dumps(result)))
    return result


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
