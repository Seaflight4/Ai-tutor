"""Generate a sample physics problem image for E2E testing."""

from __future__ import annotations

import io

from PIL import Image, ImageDraw, ImageFont


def make_sample_problem() -> bytes:
    img = Image.new("RGB", (700, 260), color="white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
        bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
        bold = font
    draw.text((20, 20), "Physics Problem", fill="black", font=bold)
    draw.text(
        (20, 70),
        "A 2.0 kg block slides down a frictionless ramp",
        fill="black",
        font=font,
    )
    draw.text(
        (20, 100),
        "from a height of 5.0 m. Using conservation of",
        fill="black",
        font=font,
    )
    draw.text(
        (20, 130),
        "energy, find the speed of the block at the",
        fill="black",
        font=font,
    )
    draw.text(
        (20, 160),
        "bottom of the ramp. (g = 9.8 m/s^2)",
        fill="black",
        font=font,
    )
    draw.text((20, 200), "Hint: PE_top = KE_bottom", fill="gray", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/problem.png"
    data = make_sample_problem()
    with open(path, "wb") as f:
        f.write(data)
    print(f"wrote {len(data)} bytes to {path}")
