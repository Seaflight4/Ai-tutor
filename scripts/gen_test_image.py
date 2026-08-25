"""Generate printed physics problem images for the test loop.

Outputs PNGs with realistic physics problem text rendered using DejaVuSans.
Images are saved to data/ (project-relative) so student subagents can't
clobber them.

Image A: ball/ground elastic collision (concepts: momentum, elastic collision)
Image B: 1D cart collision (concepts: momentum, impulse) — shares "momentum"
with image A so the cross-session learning record can be exercised.

    python -m scripts.gen_test_image           # generate both
    python -m scripts.gen_test_image --only a   # generate only image A
    python -m scripts.gen_test_image --only b    # generate only image B
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _PROJECT_ROOT / "data"

IMAGES: dict[str, dict] = {
    "a": {
        "path": DATA_DIR / "test_problem.png",
        "text": [
            "A ball of mass 100 g hits the ground at an angle of 45 degrees",
            "and a speed of 10 m/s. The collision is elastic, and the ball",
            "makes contact with the ground for a total time of 100 ms.",
            "",
            "What is the magnitude of the average net force on the ball",
            "during the collision?",
            "",
            "A. 7.1 N",
            "B. 10 N",
            "C. 14 N",
            "D. 20 N",
        ],
    },
    "b": {
        "path": DATA_DIR / "test_problem_2.png",
        "text": [
            "A 2 kg cart moving at 3 m/s collides with a stationary 1 kg cart",
            "on a frictionless track. After the collision, the 2 kg cart moves",
            "at 1 m/s in the same direction.",
            "",
            "What is the impulse delivered to the 1 kg cart during the",
            "collision?",
            "",
            "A. 1 N·s",
            "B. 2 N·s",
            "C. 4 N·s",
            "D. 6 N·s",
        ],
    },
}


def _generate(spec: dict) -> None:
    path: Path = spec["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (800, 400), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28
        )
    except OSError:
        font = ImageFont.load_default()

    y = 30
    for line in spec["text"]:
        draw.text((40, y), line, fill="black", font=font)
        y += 35

    img.save(path, format="PNG")
    print(f"saved {path} ({path.stat().st_size} bytes)", file=sys.stderr)


def main() -> None:
    args = sys.argv[1:]
    if "--only" in args:
        idx = args.index("--only")
        key = args[idx + 1].lower()
        if key not in IMAGES:
            print(f"unknown image key: {key} (expected: {list(IMAGES)})", file=sys.stderr)
            sys.exit(2)
        _generate(IMAGES[key])
        return
    for spec in IMAGES.values():
        _generate(spec)


if __name__ == "__main__":
    main()
