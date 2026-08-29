"""Download the PD *Lawrence of Arabia* trailer and cut the committed clip.

Not run at render. Needs ffmpeg and network.

    .venv-tensor-factorizations/bin/python \\
        posts/uses-of-tensor-factorizations/src/fetch_media.py
"""

from __future__ import annotations

import subprocess
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent / "media"
CACHE = Path("/tmp/lawrence-arabia-trailer.webm")
COMMONS = (
    "https://upload.wikimedia.org/wikipedia/commons/f/fe/"
    "Lawrence_Of_Arabia_%281962%29_-_Trailer.webm"
)
# Train explosion in the 1962 theatrical trailer, then the cut that follows.
START = "00:00:55"
DURATION = "5.0"
FPS = "12"
FRAME_W, FRAME_H = 160, 120
# Trailer is 1920×800; centre-crop 4:3 so the tensor is picture, not bars.
CROP_4X3 = "crop=min(iw\\,ih*4/3):ih:(iw-ow)/2:0"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"cache hit {dest} ({dest.stat().st_size:,} bytes)")
        return
    req = urllib.request.Request(
        url, headers={"User-Agent": "ml-blog/uses-of-tensor-factorizations"}
    )
    print(f"fetch {url}")
    with urllib.request.urlopen(req) as resp, dest.open("wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    print(f"wrote {dest} ({dest.stat().st_size:,} bytes)")


def ffmpeg(*args: str) -> None:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args]
    subprocess.run(cmd, check=True)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    download(COMMONS, CACHE)
    still = ROOT / "still.png"
    frames_dir = ROOT / "_frames"
    wav = ROOT / "clip.wav"
    clip = ROOT / "clip.mp4"
    ffmpeg(
        "-i", str(CACHE), "-ss", START, "-vframes", "1",
        "-vf", f"{CROP_4X3},scale=320:240", str(still),
    )
    frames_dir.mkdir(exist_ok=True)
    for old in frames_dir.glob("*.png"):
        old.unlink()
    ffmpeg(
        "-i", str(CACHE), "-ss", START, "-t", DURATION,
        "-vf", f"fps={FPS},{CROP_4X3},scale={FRAME_W}:{FRAME_H}",
        str(frames_dir / "f%03d.png"),
    )
    ffmpeg(
        "-i", str(CACHE), "-ss", START, "-t", DURATION,
        "-ac", "1", "-ar", "8000", "-vn", str(wav),
    )
    ffmpeg(
        "-i", str(CACHE), "-ss", START, "-t", DURATION,
        "-vf", f"{CROP_4X3},scale=320:240",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ac", "1", "-b:a", "64k",
        str(clip),
    )
    frame_paths = sorted(frames_dir.glob("*.png"))
    stack = np.stack(
        [np.asarray(Image.open(p).convert("RGB")) for p in frame_paths], axis=3
    )
    np.save(ROOT / "frames.npy", stack.astype(np.uint8))
    for old in frame_paths:
        old.unlink()
    frames_dir.rmdir()
    print(
        f"still {still.stat().st_size:,} B; frames {stack.shape}; "
        f"wav {wav.stat().st_size:,} B; mp4 {clip.stat().st_size:,} B"
    )


if __name__ == "__main__":
    main()
