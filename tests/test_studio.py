"""Run inside the app image: python -m unittest discover -s tests -v."""
import base64
import io
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="frame-tests-")
os.environ["XAI_API_KEY"] = ""
os.environ["NVIDIA_API_KEY"] = ""
os.environ["APP_PASSWORD"] = ""

from fastapi.testclient import TestClient
from PIL import Image
from backend import app as studio
from backend.providers import ProviderError


def image_data(color):
    image = Image.new("RGBA", (480, 480), color)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()


def payload(**extra):
    return {"title": "Test poster", "mode": "local", "preset": "zoom", "duration": 5,
            "aspect": "1:1", "resolution": "480p", "prompt": "Subtle cinematic motion",
            "image": image_data("orange"), "overlay": image_data((0, 0, 0, 0)), **extra}


class StudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.old_data = studio.DATA
        studio.DATA = Path(self.temp.name)
        self.client = TestClient(studio.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        studio.DATA = self.old_data
        self.temp.cleanup()

    def test_local_mp4_export_and_cache(self):
        result = self.client.post("/api/clips", json=payload())
        self.assertEqual(result.status_code, 202, result.text)
        job = result.json()
        for _ in range(120):
            job = self.client.get("/api/clips/" + job["id"]).json()
            if job["status"] in ("completed", "failed"):
                break
            time.sleep(.1)
        self.assertEqual(job["status"], "completed", job)
        video = self.client.get(job["video_url"])
        self.assertEqual(video.status_code, 200)
        self.assertEqual(video.headers["content-type"], "video/mp4")
        self.assertIn(b"ftyp", video.content[:32])
        cached = self.client.post("/api/clips", json=payload()).json()
        self.assertEqual(cached["id"], job["id"])
        self.assertNotIn("spec", cached)
        self.assertNotIn("fingerprint", cached)
        ranged = self.client.get(job["video_url"], headers={"Range": "bytes=0-31"})
        self.assertEqual(ranged.status_code, 206)
        self.assertEqual(len(ranged.content), 32)

    def test_invalid_image_and_duration(self):
        response = self.client.post("/api/clips", json=payload(image="data:image/png;base64,bad!"))
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.client.get("/api/clips").json(), [])
        response = self.client.post("/api/clips", json=payload(duration=100))
        self.assertEqual(response.status_code, 422)

    def test_grok_requires_key_and_cost_confirmation(self):
        response = self.client.post("/api/clips", json=payload(mode="grok"))
        self.assertEqual(response.status_code, 503)
        with patch.dict(os.environ, {"XAI_API_KEY": "test-key"}):
            response = self.client.post("/api/clips", json=payload(mode="grok"))
            self.assertEqual(response.status_code, 422)

    def test_daily_budget_blocks_upstream_call(self):
        with patch.dict(os.environ, {"XAI_API_KEY": "test-key", "MAX_DAILY_USD": "0.05"}):
            with patch.object(studio.video_provider, "submit", new_callable=AsyncMock) as submit:
                response = self.client.post("/api/clips", json=payload(mode="grok", confirm_cost=True))
                self.assertEqual(response.status_code, 429)
                submit.assert_not_called()

    def test_failed_submission_never_automatically_retries(self):
        with patch.dict(os.environ, {"XAI_API_KEY": "test-key"}):
            with patch.object(studio.video_provider, "submit", new_callable=AsyncMock, side_effect=ProviderError("Provider rate limit reached")) as submit:
                response = self.client.post("/api/clips", json=payload(mode="grok", confirm_cost=True))
                job = response.json()
                for _ in range(30):
                    job = self.client.get("/api/clips/" + job["id"]).json()
                    if job["status"] == "failed":
                        break
                    time.sleep(.05)
                self.assertEqual(job["status"], "failed")
                self.assertEqual(submit.await_count, 1)
                self.assertNotIn("request_id", job)

    def test_cross_origin_write_is_rejected(self):
        response = self.client.post("/api/clips", json=payload(), headers={"Origin": "https://untrusted.example"})
        self.assertEqual(response.status_code, 403)

    def test_password_protects_ui_and_api(self):
        with patch.dict(os.environ, {"APP_PASSWORD": "test-password"}):
            self.assertEqual(self.client.get("/").status_code, 401)
            self.assertEqual(self.client.get("/api/clips").status_code, 401)
            self.assertEqual(self.client.get("/api/config", auth=("studio", "test-password")).status_code, 200)
            self.assertEqual(self.client.get("/health").status_code, 200)

    def test_nvidia_prompt_cache(self):
        with patch.object(studio.NvidiaTextProvider, "motion_prompt", new_callable=AsyncMock, return_value={"prompt": "Slow camera drift around the sculpture", "provider": "nvidia"}) as prompt:
            body = {"brief": "A chrome sculpture on orange", "preset": "zoom"}
            first = self.client.post("/api/prompt", json=body)
            second = self.client.post("/api/prompt", json=body)
            self.assertEqual(first.status_code, 200)
            self.assertTrue(second.json()["cached"])
            self.assertEqual(prompt.await_count, 1)


if __name__ == "__main__":
    unittest.main()
