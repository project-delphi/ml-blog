"""Synthesise the post's two demo clips with Kokoro-82M.

A build tool, not part of the post's execution environment — the same
arrangement as ``make_cover.py``. The post's own kernel is stdlib-only, so
Kokoro (and its ~327MB of weights) lives in a borrowed venv instead of the
post's ``requirements.txt``. Versions are frozen in
``src/requirements-audio.txt``. Run from the post directory:

    ../../.venv-kokoro/bin/python src/make_audio.py

The two clips are the same assistant answering the same caller, differing only
in what long-term memory supplied: the first knows nothing, the second has the
name from a memory file and the voice pack from a key-value lookup. That second
voice ID, ``bm_george``, is the value the post's key-value store returns.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import soundfile as sf
from huggingface_hub import list_repo_files
from kokoro import KPipeline

REPO = "hexgrad/Kokoro-82M"
OUT = Path(__file__).resolve().parent.parent / "audio"
SAMPLE_RATE = 24_000

# (filename stem, Kokoro voice pack, text). Voice packs are prefixed by language
# and gender: `a` American English, `b` British; `f` female, `m` male.
CLIPS = [
    (
        "session-1-default",
        "af_heart",
        "Nice to meet you. How can I help?",
    ),
    (
        "session-2-remembered",
        "bm_george",
        "Morning, Alex. Ready to talk about tomorrow's ride?",
    ),
]


def available_voices() -> set[str]:
    """List the voice packs published in the Kokoro repository.

    Returns:
        Voice pack names, without the ``voices/`` prefix or ``.pt`` suffix.
    """
    return {f[len("voices/") : -len(".pt")] for f in list_repo_files(REPO) if f.startswith("voices/")}


def synthesise(pipeline: KPipeline, voice: str, text: str, wav: Path) -> None:
    """Write one clip to ``wav``.

    Args:
        pipeline: A Kokoro pipeline for the voice's language code.
        voice: Voice pack name, e.g. ``af_heart``.
        text: What to say.
        wav: Destination path.

    Raises:
        RuntimeError: If Kokoro yields no audio for the text.
    """
    chunks = [audio for _, _, audio in pipeline(text, voice=voice)]
    if not chunks:
        raise RuntimeError(f"kokoro produced no audio for {voice}: {text!r}")
    sf.write(wav, [sample for chunk in chunks for sample in chunk], SAMPLE_RATE)


def main() -> int:
    """Write every clip in ``CLIPS`` as an MP3 under ``audio/``.

    Returns:
        Process exit code.
    """
    voices = available_voices()
    missing = {voice for _, voice, _ in CLIPS} - voices
    if missing:
        raise SystemExit(f"voice packs not published in {REPO}: {sorted(missing)}")

    OUT.mkdir(exist_ok=True)
    # One pipeline per language code — the leading letter of the voice pack.
    pipelines: dict[str, KPipeline] = {}

    for stem, voice, text in CLIPS:
        pipelines.setdefault(voice[0], KPipeline(lang_code=voice[0]))
        wav, mp3 = OUT / f"{stem}.wav", OUT / f"{stem}.mp3"
        synthesise(pipelines[voice[0]], voice, text, wav)
        # 64k mono MP3 is transparent for speech and ~8x smaller than the WAV.
        # Resample to 44.1kHz on the way out: Kokoro emits 24kHz, which encodes
        # as MPEG-2 Layer III, a less universally supported profile than the
        # MPEG-1 Layer III you get at 44.1kHz. Same size at this bitrate.
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", wav,
                "-codec:a", "libmp3lame", "-b:a", "64k", "-ac", "1", "-ar", "44100", mp3,
            ],
            check=True,
        )
        wav.unlink()
        print(f"wrote {mp3.relative_to(OUT.parent)} ({mp3.stat().st_size // 1024}KB, {voice})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
