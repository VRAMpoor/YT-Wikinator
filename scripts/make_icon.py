"""Draw ytc.ico (the Control Center / desktop shortcut icon). Needs Pillow.

Usage: python scripts/make_icon.py
"""
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "ytc.ico"
SIZE = 256


def draw() -> Image.Image:
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # Dark rounded tile with a blue rim (matches the GUI theme).
    d.rounded_rectangle((8, 8, SIZE - 8, SIZE - 8), radius=52, fill="#0d1117", outline="#58a6ff", width=10)
    # Play triangle: the YouTube side.
    d.polygon([(92, 70), (92, 186), (190, 128)], fill="#f85149")
    # One text line under it: the wiki page the video is turned into.
    d.rounded_rectangle((53, 204, 203, 218), radius=7, fill="#3fb950")
    return img


def main() -> None:
    img = draw()
    img.save(OUT, format="ICO", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
