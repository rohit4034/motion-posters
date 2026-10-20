"""Provider boundary: no API keys or upstream response bodies reach clients."""
import json
import os
import time
from pathlib import Path

import httpx


class ProviderError(Exception):
    pass


def check_response(response):
    if response.status_code in (401, 403):
        raise ProviderError("The provider rejected this key or model access. Check your server configuration.")
    if response.status_code == 402:
        raise ProviderError("Your provider account needs credits. Free API requests are not guaranteed.")
    if response.status_code == 429:
        raise ProviderError("Provider rate limit reached. Wait before trying again; no automatic paid retry was made.")
    if response.is_error:
        raise ProviderError(f"Provider request failed (HTTP {response.status_code}). Check model availability and billing.")
    return response.json()


class XaiVideoProvider:
    name = "xai"

    def __init__(self):
        self.model = os.getenv("XAI_VIDEO_MODEL", "grok-imagine-video-1.5-lite")

    async def submit(self, spec, image):
        key = os.getenv("XAI_API_KEY", "").strip()
        if not key:
            raise ProviderError("Grok is not connected. Add XAI_API_KEY to the server .env and restart.")
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                "https://api.x.ai/v1/videos/generations",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": self.model, "prompt": spec["prompt"],
                      "image": {"url": image}, "duration": spec["duration"],
                      "aspect_ratio": spec["aspect"], "resolution": spec["resolution"]},
            )
            return check_response(response)

    async def status(self, request_id):
        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.get(
                f"https://api.x.ai/v1/videos/{request_id}",
                headers={"Authorization": f"Bearer {os.environ['XAI_API_KEY']}"},
            )
            return check_response(response)


class NvidiaTextProvider:
    name = "nvidia"

    async def motion_prompt(self, brief, preset):
        key = os.getenv("NVIDIA_API_KEY", "").strip()
        if not key:
            raise ProviderError("NVIDIA is not connected. Write your own motion direction or add NVIDIA_API_KEY.")
        model = os.getenv("NVIDIA_TEXT_MODEL", "nvidia/nemotron-3.5-super-vl-120b-a12b")
        start = time.monotonic()
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": model, "max_tokens": 1200, "temperature": 0.6,
                      "messages": [
                          {"role": "system", "content": "Write one image-to-video motion instruction, 40-70 words. Return only the instruction, no reasoning. Preserve the source artwork's subject, identity, colors, and composition. Do not add text, logos, objects, or cuts. Describe subtle physically plausible motion and camera movement. Never claim you saw the image; only use the user's description."},
                          {"role": "user", "content": f"Artwork description: {brief}\nMotion style: {preset}"}]},
            )
            data = check_response(response)
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        import re
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
        if not content:
            raise ProviderError("NVIDIA returned no prompt. Please write a direction manually or try again.")
        return {"prompt": content[:2000], "provider": self.name, "model": model,
                "latency_ms": round((time.monotonic() - start) * 1000), "usage": data.get("usage", {})}
