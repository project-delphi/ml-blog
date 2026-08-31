"""Fetch and cache the three datasets this post reads at render time.

Writes into posts/tensor-inverse-examples/data/ (arrays) and media/ (short
wavs the page plays). The data/ directory is carved out of the repo-wide
`data/` gitignore so a clone can rebuild the post without network access.

Not run at render. Needs network; the speech path also needs ffmpeg.

    /Users/ravikalia/Code/github.com/ml-blog/.venv-tensor-factorizations/bin/python \\
        posts/tensor-inverse-examples/src/fetch_data.py
"""

from __future__ import annotations

import gzip
import json
import subprocess
import tempfile
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import requests
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data"
MEDIA = ROOT / "media"
SEED = 7
UA = "ml-blog/tensor-inverse-examples"

# Three Wikimedia recordings of "The North Wind and the Sun". Different
# speakers, different languages — a constructed cocktail party whose mix
# we know. Commons FilePath follows the current hash.
_COMMONS = "https://commons.wikimedia.org/wiki/Special:FilePath/"
SPEECH = [
    {
        "key": "english",
        "url": _COMMONS + "Recording_of_speaker_of_British_English_(Received_Pronunciation).ogg",
        "title": "The North Wind and the Sun (British English, RP)",
        "reader": "P. Roach / International Phonetic Association; CC BY-SA 3.0",
        "start": "00:00:02",
        "page": "https://commons.wikimedia.org/wiki/File:Recording_of_speaker_of_British_English_(Received_Pronunciation).ogg",
    },
    {
        "key": "swedish",
        "url": _COMMONS + "Sv-The North Wind and the Sun.ogg",
        "title": "The North Wind and the Sun (Swedish)",
        "reader": "Wikimedia Commons; see file page",
        "start": "00:00:01",
        "page": "https://commons.wikimedia.org/wiki/File:Sv-The_North_Wind_and_the_Sun.ogg",
    },
    {
        "key": "foochow",
        "url": _COMMONS + "Cdo_northwind_sun_04.ogg",
        "title": "The North Wind and the Sun (Foochow)",
        "reader": "GnuDoyng; public domain",
        "start": "00:00:00",
        "page": "https://commons.wikimedia.org/wiki/File:Cdo_northwind_sun_04.ogg",
    },
]
SPEECH_SECONDS = 4.0
SPEECH_RATE = 8000

# CMU Arctic fallback: three speakers, three sentences, unrestricted wavs.
ARCTIC = [
    (
        "bdl",
        "arctic_a0001.wav",
        "https://www.festvox.org/cmu_arctic/cmu_us_bdl_arctic/wav/arctic_a0001.wav",
    ),
    (
        "slt",
        "arctic_a0010.wav",
        "https://www.festvox.org/cmu_arctic/cmu_us_slt_arctic/wav/arctic_a0010.wav",
    ),
    (
        "rms",
        "arctic_a0020.wav",
        "https://www.festvox.org/cmu_arctic/cmu_us_rms_arctic/wav/arctic_a0020.wav",
    ),
]

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
N_USERS = 80
N_MOVIES = 80

DRYER_URL = (
    "https://ftp.esat.kuleuven.be/pub/SISTA/data/process_industry/dryer2.dat.gz"
)
DRYER_TXT = (
    "https://ftp.esat.kuleuven.be/pub/SISTA/data/process_industry/dryer2.txt"
)
DRYER_FALLBACK = (
    "https://ftp.esat.kuleuven.be/pub/SISTA/data/process_industry/"
    "glassfurnace.dat.gz"
)


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"kept  {dest} ({dest.stat().st_size:,} bytes)")
        return dest
    print(f"GET   {url}")
    # files.grouplens.org presents an expired certificate (checked 2026-08-31).
    verify = "grouplens.org" not in url
    resp = requests.get(
        url,
        headers={"User-Agent": UA},
        timeout=120,
        allow_redirects=True,
        verify=verify,
    )
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"wrote {dest} ({dest.stat().st_size:,} bytes)")
    return dest


def _ffmpeg(*args: str) -> None:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args]
    subprocess.run(cmd, check=True)


def _write_wav(path: Path, rate: int, signal: np.ndarray) -> None:
    peak = np.max(np.abs(signal)) + 1e-12
    pcm = np.int16(np.clip(signal / peak, -1.0, 1.0) * 32767)
    wavfile.write(path, rate, pcm)


def _mix_matrix(rng: np.random.Generator) -> np.ndarray:
    """A well-conditioned 3x3 mix so the inverse exists and is not trivial."""
    A = rng.normal(size=(3, 3))
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    if np.linalg.cond(A) > 8:
        A = np.array(
            [[0.8, 0.5, 0.2], [0.3, 0.9, 0.4], [0.25, 0.35, 0.85]],
            dtype=float,
        )
    return A


def _fetch_librivox(tmp: Path) -> tuple[np.ndarray, list[str], str]:
    clips = []
    labels = []
    for spec in SPEECH:
        raw = tmp / f"{spec['key']}.mp3"
        wav = tmp / f"{spec['key']}.wav"
        _download(spec["url"], raw)
        _ffmpeg(
            "-i",
            str(raw),
            "-ss",
            spec["start"],
            "-t",
            str(SPEECH_SECONDS),
            "-ac",
            "1",
            "-ar",
            str(SPEECH_RATE),
            str(wav),
        )
        rate, sig = wavfile.read(wav)
        assert rate == SPEECH_RATE
        clips.append(sig.astype(float))
        labels.append(spec["title"])
    n = min(c.size for c in clips)
    sources = np.stack([c[:n] for c in clips])
    return sources, labels, "commons"


def _fetch_arctic(tmp: Path) -> tuple[np.ndarray, list[str], str]:
    clips = []
    labels = []
    for speaker, name, url in ARCTIC:
        dest = tmp / name
        _download(url, dest)
        rate, sig = wavfile.read(dest)
        if sig.ndim > 1:
            sig = sig.mean(axis=1)
        if rate != SPEECH_RATE:
            # Linear resample; Arctic is 16 kHz.
            t_old = np.linspace(0.0, 1.0, sig.size, endpoint=False)
            n_new = int(sig.size * SPEECH_RATE / rate)
            t_new = np.linspace(0.0, 1.0, n_new, endpoint=False)
            sig = np.interp(t_new, t_old, sig.astype(float))
        clips.append(sig.astype(float))
        labels.append(f"CMU Arctic {speaker} / {name}")
    n = min(c.size for c in clips)
    sources = np.stack([c[:n] for c in clips])
    return sources, labels, "cmu-arctic"


def fetch_speech() -> Path:
    out = OUT / "speech.npz"
    if out.exists() and (MEDIA / "mix.wav").exists():
        print(f"kept  {out} ({out.stat().st_size:,} bytes)")
        return out
    MEDIA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        try:
            sources, labels, origin = _fetch_librivox(tmp)
        except Exception as err:
            print(f"      commons failed ({err}); trying CMU Arctic")
            sources, labels, origin = _fetch_arctic(tmp)
    # Unit-variance sources, then a known mix. The mix is constructed; the
    # waveforms are not.
    sources = sources - sources.mean(axis=1, keepdims=True)
    sources = sources / (sources.std(axis=1, keepdims=True) + 1e-12)
    mix_mat = _mix_matrix(rng)
    mixed = mix_mat @ sources
    for i, sig in enumerate(sources):
        _write_wav(MEDIA / f"source-{i}.wav", SPEECH_RATE, sig)
    for i, sig in enumerate(mixed):
        _write_wav(MEDIA / f"mix-{i}.wav", SPEECH_RATE, sig)
    _write_wav(MEDIA / "mix.wav", SPEECH_RATE, mixed.mean(axis=0))
    np.savez_compressed(
        out,
        sources=sources.astype(np.float32),
        mixed=mixed.astype(np.float32),
        mix_matrix=mix_mat.astype(np.float32),
        rate=np.array(SPEECH_RATE),
        labels=np.array(labels),
        origin=np.array(origin),
        seed=np.array(SEED),
        retrieved=np.array(date.today().isoformat()),
    )
    print(f"wrote {out} {sources.shape} from {origin}")
    return out


def fetch_movielens() -> Path:
    out = OUT / "movielens.npz"
    if out.exists():
        print(f"kept  {out} ({out.stat().st_size:,} bytes)")
        return out
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_s:
        zpath = Path(tmp_s) / "ml-100k.zip"
        _download(MOVIELENS_URL, zpath)
        with zipfile.ZipFile(zpath) as zf:
            ratings = zf.read("ml-100k/u.data").decode("ascii")
            items = zf.read("ml-100k/u.item").decode("latin-1")
    rows = [line.split("\t") for line in ratings.splitlines() if line]
    user = np.array([int(r[0]) for r in rows])
    movie = np.array([int(r[1]) for r in rows])
    rating = np.array([float(r[2]) for r in rows])
    ts = np.array([int(r[3]) for r in rows])
    titles = {}
    for line in items.splitlines():
        if not line:
            continue
        parts = line.split("|")
        titles[int(parts[0])] = parts[1]
    # Most-active users and most-rated movies, then month bins.
    user_counts = {u: n for u, n in zip(*np.unique(user, return_counts=True))}
    movie_counts = {m: n for m, n in zip(*np.unique(movie, return_counts=True))}
    keep_u = np.array(
        sorted(user_counts, key=user_counts.get, reverse=True)[:N_USERS]
    )
    keep_m = np.array(
        sorted(movie_counts, key=movie_counts.get, reverse=True)[:N_MOVIES]
    )
    months = np.array(
        [
            datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m")
            for t in ts
        ]
    )
    month_levels = np.array(sorted(set(months)))
    u_index = {int(u): i for i, u in enumerate(keep_u)}
    m_index = {int(m): i for i, m in enumerate(keep_m)}
    t_index = {m: i for i, m in enumerate(month_levels)}
    sums = np.zeros((N_USERS, N_MOVIES, month_levels.size))
    counts = np.zeros_like(sums)
    for u, m, r, month in zip(user, movie, rating, months):
        if int(u) not in u_index or int(m) not in m_index:
            continue
        i, j, k = u_index[int(u)], m_index[int(m)], t_index[month]
        sums[i, j, k] += r
        counts[i, j, k] += 1
    mean = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    title_list = np.array([titles[int(m)] for m in keep_m])
    np.savez_compressed(
        out,
        ratings=mean.astype(np.float32),
        counts=counts.astype(np.int16),
        users=keep_u.astype(np.int32),
        movies=keep_m.astype(np.int32),
        titles=title_list,
        months=month_levels,
        source=np.array(MOVIELENS_URL),
        retrieved=np.array(date.today().isoformat()),
    )
    n_obs = int(np.isfinite(mean).sum())
    print(f"wrote {out} {mean.shape}, {n_obs} observed cells")
    return out


def _parse_daisy(raw: bytes) -> np.ndarray:
    text = gzip.decompress(raw).decode("ascii", errors="ignore")
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            continue
    if not rows:
        raise RuntimeError("no numeric rows in DaISy file")
    return np.asarray(rows, dtype=float)


def fetch_dryer() -> Path:
    out = OUT / "dryer.npz"
    if out.exists():
        print(f"kept  {out} ({out.stat().st_size:,} bytes)")
        return out
    OUT.mkdir(parents=True, exist_ok=True)
    origin = "dryer"
    try:
        with tempfile.TemporaryDirectory() as tmp_s:
            dest = Path(tmp_s) / "dryer2.dat.gz"
            _download(DRYER_URL, dest)
            table = _parse_daisy(dest.read_bytes())
        # dryer.dat: col 0 index, cols 1-3 inputs, cols 4-6 outputs.
        if table.shape[1] >= 7:
            U, Y = table[:, 1:4], table[:, 4:7]
        else:
            raise RuntimeError(f"unexpected dryer width {table.shape}")
        inputs = np.array(["fuel flow", "exhaust fan", "raw-material flow"])
        outputs = np.array(["dry-bulb temp", "wet-bulb temp", "moisture"])
        dt = 10.0
        cite = "96-016 industrial dryer (Cambridge Control Ltd)"
    except (OSError, RuntimeError) as err:
        print(f"      dryer failed ({err}); trying glass furnace")
        origin = "glassfurnace"
        with tempfile.TemporaryDirectory() as tmp_s:
            dest = Path(tmp_s) / "glassfurnace.dat.gz"
            _download(DRYER_FALLBACK, dest)
            table = _parse_daisy(dest.read_bytes())
        # Glass furnace is typically 3 inputs + 6 outputs; take the first 3 out.
        U, Y = table[:, 1:4], table[:, 4:7]
        inputs = np.array(["input 1", "input 2", "input 3"])
        outputs = np.array(["output 1", "output 2", "output 3"])
        dt = 1.0
        cite = "96-002 glass furnace (Philips)"
    np.savez_compressed(
        out,
        U=U.astype(np.float32),
        Y=Y.astype(np.float32),
        inputs=inputs,
        outputs=outputs,
        dt=np.array(dt),
        origin=np.array(origin),
        cite=np.array(cite),
        source=np.array(DRYER_URL if origin == "dryer" else DRYER_FALLBACK),
        retrieved=np.array(date.today().isoformat()),
    )
    print(f"wrote {out} U{U.shape} Y{Y.shape} ({cite})")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MEDIA.mkdir(parents=True, exist_ok=True)
    fetch_speech()
    fetch_movielens()
    fetch_dryer()
    manifest = {
        "speech": {
            "clips": [
                {
                    "url": s["url"],
                    "title": s["title"],
                    "start": s["start"],
                    "page": s.get("page", s["url"]),
                }
                for s in SPEECH
            ],
            "seconds": SPEECH_SECONDS,
            "rate": SPEECH_RATE,
            "note": "Three Wikimedia recordings of The North Wind and the Sun, mixed with a known 3x3.",
        },
        "movielens": {
            "url": MOVIELENS_URL,
            "users": N_USERS,
            "movies": N_MOVIES,
            "note": "MovieLens 100K crop: most-active users x most-rated movies x month.",
        },
        "dryer": {
            "url": DRYER_URL,
            "description": DRYER_TXT,
            "cite": (
                "De Moor B.L.R. (ed.), DaISy: Database for the Identification "
                "of Systems, ESAT/STADIUS, KU Leuven. Dataset 96-016."
            ),
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (MEDIA / "attribution.txt").write_text(
        "Speech: Wikimedia Commons recordings of The North Wind and the Sun "
        "(British English RP, P. Roach / IPA, CC BY-SA 3.0; Swedish; "
        "Foochow, GnuDoyng, public domain). "
        "Fallback: CMU Arctic (Festvox, unrestricted).\n"
        "Ratings: MovieLens 100K, GroupLens Research "
        "(Harper and Konstan, ACM TiiS 2015).\n"
        "Dryer: DaISy 96-016 industrial dryer, Cambridge Control Ltd, "
        "contributed by J. M. Maciejowski; "
        "De Moor B.L.R. (ed.), DaISy, ESAT/STADIUS, KU Leuven.\n"
    )


if __name__ == "__main__":
    main()
