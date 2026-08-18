"""Convert the source HEIC frames into the committed half-resolution JPEGs.

The originals are 4032x3024 iPhone HEICs that live outside the repo. Everything
downstream -- the segmentation, the landmarks in ``landmarks.csv``, the figures
in the post -- works in the coordinate frame this script writes, so there is
exactly one pixel grid in the repository and no rescaling to get wrong later.

Not run by the render. Regenerate the photos by hand::

    .venv-kite/bin/python posts/kite-shape-analysis/src/prepare_photos.py
"""

from __future__ import annotations

from pathlib import Path

import pillow_heif
from PIL import Image

pillow_heif.register_heif_opener()

SOURCE = Path.home() / "Downloads"
OUT = Path(__file__).resolve().parent.parent / "photos"

# The five frames of the kite festival, in the order the camera took them.
# IMG_0555 (1).HEIC is a byte-identical re-download of the first and is skipped.
FRAMES = ["IMG_0555", "IMG_0556", "IMG_0557", "IMG_0558", "IMG_0559"]

# Half of 4032x3024. Big enough that the kite is still 250-310 px across --
# landmark noise stays small next to the sail -- and small enough to commit.
SCALE = 2


def main() -> int:
    """Write ``photos/kite-01.jpg`` .. ``kite-05.jpg``.

    Returns
    -------
        Process exit code.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    for index, stem in enumerate(FRAMES, start=1):
        source = SOURCE / f"{stem}.HEIC"
        if not source.exists():
            print(f"missing {source}")
            return 1
        # `exif_transpose` is not needed: these frames carry no rotation tag,
        # and applying one would silently move every landmark.
        image = Image.open(source).convert("RGB")
        width, height = image.size
        resized = image.resize((width // SCALE, height // SCALE), Image.LANCZOS)
        target = OUT / f"kite-{index:02d}.jpg"
        resized.save(target, quality=88, optimize=True)
        print(f"{source.name} -> {target.name}  {resized.size[0]}x{resized.size[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
