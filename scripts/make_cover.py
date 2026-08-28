#!/usr/bin/env python3
"""Write posts/<slug>/cover.png from an in-post visual or Wikimedia Commons.

Covers are content-derived: copy the figure that carries the post's main claim,
or — if the post has no raster of its own — a license-safe Commons file. Never
draw the old purple title card.

Run with Pillow (and PyYAML, for --all) from the repo root::

    uv run --with pillow --with pyyaml python scripts/make_cover.py \\
        posts/volcano-plots --source \\
        _freeze/posts/volcano-plots/index/figure-html/fig-airway-output-1.png

    uv run --with pillow --with pyyaml python scripts/make_cover.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = Path(__file__).resolve().parent / "cover_sources.yml"
WIDTH, HEIGHT = 1200, 630
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "ml-blog-cover-bot/1.0 (https://github.com/project-delphi/ml-blog)"

# LicenseShortName values Wikimedia returns. NC is rejected; ND is accepted
# because we only resize, we do not crop out the work or add a derivative scene.
ALLOWED_LICENSE_PREFIXES = (
    "public domain",
    "pd",
    "cc0",
    "cc by",
    "cc-by",
    "creative commons attribution",
    "gfdl",
)


def load_yaml(path: Path) -> dict[str, dict[str, object]]:
    """Read the slug → source map.

    Args:
        path: Path to cover_sources.yml.

    Returns:
        Mapping of post slug to a dict with skip/source/commons keys.
    """
    import yaml  # noqa: PLC0415  # optional extra via uv run --with pyyaml

    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must be a mapping of slug → entry")
    return data


def open_rgba(path: Path):
    """Open a raster, taking the first frame of an animated source.

    Args:
        path: Image file on disk.

    Returns:
        An RGBA Pillow image.
    """
    from PIL import Image  # noqa: PLC0415  # optional extra via uv run --with pillow

    image = Image.open(path)
    image.seek(0)
    return image.convert("RGBA")


def fit_cover(image) -> object:
    """Scale an image to fill WIDTH×HEIGHT and center-crop the overflow.

    Args:
        image: A Pillow image.

    Returns:
        A new RGB image of exactly WIDTH by HEIGHT.
    """
    from PIL import Image  # noqa: PLC0415

    src_w, src_h = image.size
    if src_w == 0 or src_h == 0:
        raise SystemExit("source image has zero size")
    scale = max(WIDTH / src_w, HEIGHT / src_h)
    new_w = max(WIDTH, round(src_w * scale))
    new_h = max(HEIGHT, round(src_h * scale))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = (new_w - WIDTH) // 2
    top = (new_h - HEIGHT) // 2
    cropped = resized.crop((left, top, left + WIDTH, top + HEIGHT))
    background = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
    background.paste(cropped, mask=cropped.split()[-1] if cropped.mode == "RGBA" else None)
    return background


def write_cover(slug: str, source: Path) -> Path:
    """Fit `source` into posts/<slug>/cover.png.

    Args:
        slug: Post directory name.
        source: Raster to copy from.

    Returns:
        Path of the written cover.
    """
    if not source.is_file():
        raise SystemExit(f"source not found: {source}")
    out = ROOT / "posts" / slug / "cover.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fitted = fit_cover(open_rgba(source))
    fitted.save(out, "PNG", optimize=True)
    return out


def license_allowed(short_name: str) -> bool:
    """Return True if a Commons LicenseShortName is PD or CC-BY (not NC).

    Args:
        short_name: The LicenseShortName Wikimedia reports.

    Returns:
        Whether we will commit the file as a cover.
    """
    lowered = short_name.strip().lower()
    if "-nc" in lowered or "noncommercial" in lowered or "non-commercial" in lowered:
        return False
    return any(lowered.startswith(prefix) or prefix in lowered for prefix in ALLOWED_LICENSE_PREFIXES)


def commons_info(filename: str) -> dict[str, str]:
    """Resolve a Commons File: title to a raster URL and license metadata.

    Args:
        filename: Commons file title, with or without the File: prefix.

    Returns:
        url, page, license, artist, and title.
    """
    title = filename if filename.startswith("File:") else f"File:{filename}"
    query = urllib.parse.urlencode(
        {
            "action": "query",
            "titles": title,
            "prop": "imageinfo",
            "iiprop": "url|mime|extmetadata",
            "iiurlwidth": str(WIDTH),
            "format": "json",
        }
    )
    request = urllib.request.Request(f"{COMMONS_API}?{query}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode())
    pages = payload.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})
    if int(page.get("pageid", -1)) < 0:
        raise SystemExit(f"Commons file not found: {title}")
    info = (page.get("imageinfo") or [None])[0]
    if not info:
        raise SystemExit(f"Commons file has no imageinfo: {title}")
    meta = info.get("extmetadata") or {}
    license_name = (meta.get("LicenseShortName") or {}).get("value") or ""
    if not license_allowed(license_name):
        raise SystemExit(f"Commons license not allowed for {title}: {license_name!r}")
    url = info.get("thumburl") or info.get("url")
    if not url:
        raise SystemExit(f"Commons file has no download URL: {title}")
    return {
        "url": url,
        "page": info.get("descriptionshorturl") or info.get("descriptionurl") or "",
        "license": license_name,
        "artist": (meta.get("Artist") or {}).get("value") or "",
        "credit": (meta.get("Credit") or {}).get("value") or "",
        "title": page.get("title") or title,
    }


def download(url: str, dest: Path) -> None:
    """Fetch `url` to `dest` with the Wikimedia user agent.

    Args:
        url: Direct file or thumbnail URL.
        dest: Local path to write.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        dest.write_bytes(response.read())


def strip_tags(html: str) -> str:
    """Drop simple HTML tags from Commons metadata fields.

    Args:
        html: Artist/credit snippet, which Commons often wraps in links.

    Returns:
        Plain text.
    """
    import re  # noqa: PLC0415

    return re.sub(r"<[^>]+>", "", html).strip()


def write_attribution(slug: str, info: dict[str, str]) -> Path:
    """Write posts/<slug>/cover-source.txt from Commons metadata.

    Args:
        slug: Post directory name.
        info: Result of commons_info.

    Returns:
        Path of the attribution file.
    """
    out = ROOT / "posts" / slug / "cover-source.txt"
    artist = strip_tags(info["artist"]) or "(see Commons page)"
    lines = [
        "cover.png",
        "--------",
        f"Title:    {info['title']}",
        f"Artist:   {artist}",
        f"Source:   Wikimedia Commons",
        f"          {info['page']}",
        f"          Direct file: {info['url']}",
        f"License:  {info['license']}",
        "Downloaded via the Wikimedia Commons API; license metadata",
        "(LicenseShortName) verified at download time.",
        "",
    ]
    out.write_text("\n".join(lines))
    return out


def process_source(slug: str, relpath: str) -> None:
    """Copy a repo-relative raster onto cover.png.

    Args:
        slug: Post directory name.
        relpath: Path relative to the repo root.
    """
    source = (ROOT / relpath).resolve()
    if not str(source).startswith(str(ROOT)):
        raise SystemExit(f"source escapes the repo: {relpath}")
    out = write_cover(slug, source)
    print(f"wrote {out.relative_to(ROOT)} from {relpath}")


def process_commons(slug: str, filename: str) -> None:
    """Download a Commons file, fit it, and record attribution.

    Args:
        slug: Post directory name.
        filename: Commons File: title.
    """
    info = commons_info(filename)
    tmp = ROOT / "posts" / slug / ".cover-download"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        download(info["url"], tmp)
        out = write_cover(slug, tmp)
    finally:
        if tmp.exists():
            tmp.unlink()
    attr = write_attribution(slug, info)
    print(f"wrote {out.relative_to(ROOT)} from Commons {info['title']}")
    print(f"wrote {attr.relative_to(ROOT)} ({info['license']})")


def process_entry(slug: str, entry: dict[str, object], *, only: str | None) -> None:
    """Dispatch one YAML entry.

    Args:
        slug: Post directory name.
        entry: skip/source/commons mapping.
        only: If set, skip entries that are not this kind (`source` or `commons`).
    """
    if entry.get("skip"):
        print(f"skip {slug}: cover is produced by the post itself")
        return
    source = entry.get("source")
    commons = entry.get("commons")
    if source and commons:
        raise SystemExit(f"{slug}: set source or commons, not both")
    if source:
        if only and only != "source":
            return
        process_source(slug, str(source))
        return
    if commons:
        if only and only != "commons":
            return
        process_commons(slug, str(commons))
        return
    raise SystemExit(f"{slug}: entry needs skip, source, or commons")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Optional argument list for tests; defaults to sys.argv.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "post",
        nargs="?",
        help="posts/<slug> or <slug> (required unless --all)",
    )
    parser.add_argument("--source", help="repo-relative path of an in-post raster")
    parser.add_argument("--commons", help="Wikimedia Commons File: title")
    parser.add_argument(
        "--all",
        action="store_true",
        help="process every entry in scripts/cover_sources.yml",
    )
    parser.add_argument(
        "--only",
        choices=("source", "commons"),
        help="with --all, restrict to in-post copies or Commons downloads",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Optional argument list.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    if args.all:
        mapping = load_yaml(SOURCES)
        for slug, entry in mapping.items():
            if not isinstance(entry, dict):
                raise SystemExit(f"{slug}: expected a mapping")
            process_entry(slug, entry, only=args.only)
        return 0

    if not args.post:
        raise SystemExit("pass posts/<slug> or --all")
    slug = args.post.rstrip("/").split("/")[-1]
    if args.source:
        process_source(slug, args.source)
        return 0
    if args.commons:
        process_commons(slug, args.commons)
        return 0

    mapping = load_yaml(SOURCES)
    if slug not in mapping:
        raise SystemExit(f"{slug} is not in {SOURCES.relative_to(ROOT)}")
    process_entry(slug, mapping[slug], only=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
