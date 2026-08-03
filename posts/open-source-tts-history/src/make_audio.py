"""Synthesise one sentence with four engines spanning thirty years of TTS.

A build tool, not part of the post's execution environment — the same
arrangement as ``make_cover.py``, and as ``llm-agent-memory``'s script of the
same name. The post itself has no executable cells; it just embeds the MP3s
this writes.

Four of the five clips are generated here, on this machine. The techniques the
post describes are not museum pieces: eSpeak NG still does formant synthesis by
rule, and Flite still ships both a diphone database and a Clustergen voice. So
the 1990s and 2000s rows are reproducible rather than merely described. The
fifth clip is Tacotron 2, which cannot be run casually on a laptop, and is
mirrored from Google's own demo page (see ``TACOTRON`` below for provenance).

Build the environment and run from the post directory::

    # The clips below were built with espeak-ng 1.52.0 and flite 2.2. These two
    # determine three of the five clips as surely as the frozen venv determines
    # the fifth, so record the versions when you rebuild on newer ones.
    brew install espeak-ng flite ffmpeg
    uv venv .venv-kokoro --python 3.12   # already present in this repo
    uv pip install --python .venv-kokoro/bin/python kokoro soundfile
    # misaki's English G2P loads a spaCy pipeline that Kokoro does not pull in,
    # and `python -m spacy download` cannot install into a pip-less uv venv:
    uv pip install --python .venv-kokoro/bin/python \
      "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
    ../../.venv-kokoro/bin/python src/make_audio.py

Versions are frozen in ``src/requirements-audio.txt``. Only Kokoro needs that
venv; everything else is stdlib plus the two Homebrew binaries.
"""

from __future__ import annotations

import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
from kokoro import KPipeline

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "audio"

# The sentence every locally-generated clip says. Holding the text fixed is the
# whole point: what changes between the players is the synthesis technique, not
# the words, so differences the reader hears are attributable to the era.
LINE = "This is open source text to speech, thirty years apart."

# Kokoro's output rate. It is a model constant that the package exposes nowhere
# — not on KModel, not on KPipeline — so it has to be written down, and it is
# load-bearing: get it wrong and the clip is pitched wrong rather than failing.
# Re-check it against the model card when bumping the pinned kokoro version.
KOKORO_SAMPLE_RATE = 24_000

# The one clip we do not generate. Tacotron 2 needs a trained checkpoint Google
# never released, so the era is represented by Google's own published sample.
# "fox_period" is model output (the page's `_gt` files are the human recordings
# it asks you to tell apart; this one is not part of that quiz). Note the source
# repository carries no explicit licence — the post attributes the clip next to
# the player and links back to the demo page.
TACOTRON_URL = (
    "https://google.github.io/tacotron/publications/tacotron2/demos/fox_period.wav"
)
TACOTRON_LINE = "The quick brown fox jumps over the lazy dog."


@dataclass(frozen=True)
class Clip:
    """One player on the page.

    Attributes:
        stem: Output filename without extension; also the ordering key.
        engine: Which synthesiser produces it — dispatched on in ``render``.
        detail: Engine-specific argument (a Flite voice name, a Kokoro voice
            pack, or ``""`` where the engine takes none).
        text: What is spoken.
    """

    stem: str
    engine: str
    detail: str
    text: str


CLIPS = [
    # Formant synthesis: no recorded speech anywhere in the pipeline, just rules
    # driving a source-filter model. eSpeak NG is the living descendant.
    Clip("01-formant-espeak-ng", "espeak", "", LINE),
    # Diphone concatenation: ~1500 recorded phone-to-phone transitions, one copy
    # of each, pitch-shifted into place. kal16 is the 16kHz cut of cmu_us_kal,
    # chosen over the 8kHz `kal` so bandwidth does not get mistaken for
    # technique.
    Clip("02-diphone-flite-kal", "flite", "kal16", LINE),
    # Statistical parametric: cmu_us_slt_cg is a Clustergen voice — MCEP trees,
    # an F0 tree and a duration model, no stored waveforms. Same family as the
    # HMM-based HTS systems of the era, though not literally HTS.
    Clip("03-parametric-flite-slt", "flite", "slt", LINE),
    # Neural seq2seq + neural vocoder, mirrored rather than generated.
    Clip("04-tacotron2-google", "download", TACOTRON_URL, TACOTRON_LINE),
    # Codec era. StyleTTS 2 architecture, 82M parameters, runs on CPU.
    Clip("05-kokoro-82m", "kokoro", "af_heart", LINE),
]


def render(clip: Clip, wav: Path, pipelines: dict[str, KPipeline]) -> None:
    """Produce ``clip`` as a WAV at ``wav``.

    Args:
        clip: The clip to render.
        wav: Destination path for the intermediate WAV.
        pipelines: Cache of Kokoro pipelines, keyed by language code. Mutated.

    Raises:
        RuntimeError: If Kokoro yields no audio for the text.
        ValueError: If ``clip.engine`` is not a known engine.
    """
    if clip.engine == "espeak":
        subprocess.run(
            ["espeak-ng", "-v", "en-us", "-s", "150", "-w", str(wav), clip.text],
            check=True,
        )
    elif clip.engine == "flite":
        subprocess.run(
            ["flite", "-voice", clip.detail, "-o", str(wav), clip.text], check=True
        )
    elif clip.engine == "download":
        urllib.request.urlretrieve(clip.detail, wav)  # noqa: S310 - pinned https URL
    elif clip.engine == "kokoro":
        lang = clip.detail[0]
        pipeline = pipelines.setdefault(lang, KPipeline(lang_code=lang))
        chunks = [audio for _, _, audio in pipeline(clip.text, voice=clip.detail)]
        if not chunks:
            raise RuntimeError(f"kokoro produced no audio for {clip.detail!r}")
        sf.write(
            wav,
            [sample for chunk in chunks for sample in chunk],
            KOKORO_SAMPLE_RATE,
        )
    else:
        raise ValueError(f"unknown engine: {clip.engine!r}")


def main() -> int:
    """Write every clip in ``CLIPS`` as an MP3 under ``audio/``.

    Returns:
        Process exit code.
    """
    OUT.mkdir(exist_ok=True)
    pipelines: dict[str, KPipeline] = {}

    for clip in CLIPS:
        wav, mp3 = OUT / f"{clip.stem}.wav", OUT / f"{clip.stem}.mp3"
        render(clip, wav, pipelines)
        # 64k mono MP3 is transparent for speech and ~8x smaller than the WAV.
        # Resample to 44.1kHz on the way out: several of these engines emit 16
        # or 24kHz, which encode as MPEG-2 Layer III, a less universally
        # supported profile than the MPEG-1 Layer III you get at 44.1kHz.
        #
        # loudnorm matters more here than in a single-voice post. These five
        # engines differ in output level by more than 10 LU, and the page
        # invites the reader to click straight down the list — without
        # levelling, the comparison becomes one of volume rather than of
        # technique, and 2026 arrives painfully loud after 1996.
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
                "-af", "loudnorm=I=-18:TP=-1.5:LRA=11",
                "-codec:a", "libmp3lame", "-b:a", "64k", "-ac", "1", "-ar", "44100",
                str(mp3),
            ],
            check=True,
        )
        wav.unlink()
        print(f"wrote {mp3.relative_to(OUT.parent)} ({mp3.stat().st_size // 1024}KB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
