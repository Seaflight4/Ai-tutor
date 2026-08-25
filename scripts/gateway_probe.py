"""Probe the skainet gateway to confirm the OCR model accepts vision input.

Generates a small PNG containing a printed physics problem, sends it to the
/v1/chat/completions endpoint with the configured OCR model, and prints the
returned text. Read-only: makes one HTTP call, writes nothing to disk.

Usage:
    python scripts/gateway_probe.py
"""

from __future__ import annotations

import base64
import io
import sys

import httpx
from PIL import Image, ImageDraw, ImageFont

from app.core.config import get_settings
from app.prompts.guided_discovery import OCR_PROMPT


def make_problem_image() -> bytes:
    img = Image.new("RGB", (600, 200), color="white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
    draw.text((20, 20), "A 2 kg block slides down a frictionless", fill="black", font=font)
    draw.text((20, 60), "ramp of height 5 m. Find its speed at", fill="black", font=font)
    draw.text((20, 100), "the bottom. (g = 9.8 m/s^2)", fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main() -> int:
    settings = get_settings()
    if not settings.skainet_api_key or settings.skainet_api_key == "tngai_replace_me":
        print(
            "[probe] FAIL: SKAINET_API_KEY is not set (or is the placeholder). "
            "Set it in .env or the environment.",
            file=sys.stderr,
        )
        return 1
    print(f"[probe] base_url={settings.skainet_base_url}")
    print(f"[probe] model={settings.model_ocr}")
    image_bytes = make_problem_image()
    data_url = f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"
    payload = {
        "model": settings.model_ocr,
        "temperature": 0.0,
        "max_tokens": 500,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    }
    headers = {"Authorization": f"Bearer {settings.skainet_api_key}"}
    print("[probe] sending request...")
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"{settings.skainet_base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
    except httpx.HTTPError as exc:
        print(f"[probe] FAIL: transport error: {exc}", file=sys.stderr)
        return 1
    print(f"[probe] status: {resp.status_code}")
    if resp.status_code != 200:
        print(f"[probe] FAIL: body:\n{resp.text[:1000]}", file=sys.stderr)
        return 1
    data = resp.json()
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        print(f"[probe] FAIL: unexpected response shape:\n{data}", file=sys.stderr)
        return 1
    print("[probe] OK - extracted text:")
    print("----")
    print(text)
    print("----")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
