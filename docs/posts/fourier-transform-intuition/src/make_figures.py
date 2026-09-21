"""Build synthetic Fourier examples and check the numerical claims in the post."""

import io
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/fourier-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.io import wavfile
from scipy.signal import ShortTimeFFT, windows

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
INK, BLUE, ORANGE, GREEN = "#18344b", "#167b9c", "#c45d26", "#398166"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": INK,
        "text.color": INK,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.prop_cycle": matplotlib.cycler(color=[BLUE, ORANGE, GREEN]),
    }
)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=145, bbox_inches="tight")
    plt.close(fig)


def frame(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=85)
    buf.seek(0)
    return Image.open(buf).convert("RGB").quantize(colors=64)


def gif(frames, name, duration=130):
    frames[0].save(
        OUT / f"{name}.gif",
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )


def spectrum(x, fs):
    """One-sided amplitudes, with DC and the even-length Nyquist bin undoubled."""
    amp = np.abs(np.fft.rfft(x)) / len(x)
    amp[1 : -1 if len(x) % 2 == 0 else None] *= 2
    return np.fft.rfftfreq(len(x), 1 / fs), amp


def entropy(x):
    power = np.abs(np.fft.rfft(x - x.mean())) ** 2
    power[1:-1] *= 2  # All examples here have even length.
    power = power[1:]  # Exclude DC, retain all K non-DC bins.
    p = power / power.sum()
    positive = p > 0
    return -np.sum(p[positive] * np.log2(p[positive])) / np.log2(len(p))


def variance_figure():
    n = 256
    t = np.arange(n) / n
    slow = np.sqrt(2) * np.sin(2 * np.pi * 3 * t)
    fast = np.sqrt(2) * np.sin(2 * np.pi * 17 * t)
    shuffled = np.random.default_rng(7).permutation(slow)
    signals = [slow, fast, shuffled]
    names = ["3 cycles / second", "17 cycles / second", "Shuffled 3 Hz samples"]
    fig, axes = plt.subplots(3, 2, figsize=(10, 7.2), layout="constrained")
    for row, (x, name) in enumerate(zip(signals, names, strict=True)):
        f, a = spectrum(x, n)
        axes[row, 0].plot(t, x, lw=1.4)
        axes[row, 0].set(title=name, ylabel="Value", ylim=(-1.7, 1.7))
        axes[row, 1].plot(f, a, lw=1.5, color=ORANGE)
        axes[row, 1].set(
            title=f"Variance = {np.var(x):.2f}  |  Spectral entropy = {entropy(x):.2f}",
            ylabel="Amplitude",
            xlim=(0, 128),
            ylim=(0, 1.55),
        )
    axes[-1, 0].set_xlabel("Time (seconds)")
    axes[-1, 1].set_xlabel("Frequency (Hz)")
    fig.suptitle("Equal spread. Different rhythms.", fontsize=18)
    save(fig, "variance-frequency")
    np.testing.assert_allclose([np.var(x) for x in signals], 1)
    np.testing.assert_array_equal(np.sort(slow), np.sort(shuffled))
    print(
        "Variance examples:",
        [(s, round(entropy(x), 4)) for s, x in zip(names, signals, strict=True)],
    )


def ingredients_figure():
    fs = 128
    t = np.arange(fs) / fs
    components = [2 * np.sin(2 * np.pi * 3 * t), 0.75 * np.cos(2 * np.pi * 8 * t)]
    x = sum(components)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
    for component, label in zip(
        components,
        ["3 Hz sine; amplitude 2", "8 Hz cosine; amplitude 0.75"],
        strict=True,
    ):
        axes[0, 0].plot(t, component, label=label)
    axes[0, 0].legend(fontsize=9, loc="upper right")
    axes[0, 0].set(title="Two ingredients", ylabel="Value", xlabel="Time (seconds)")
    axes[1, 0].plot(t, x)
    axes[1, 0].set(
        title="Their sum: the observed vector", ylabel="Value", xlabel="Time (seconds)"
    )
    f, a = spectrum(x, fs)
    axes[0, 1].stem(f[:15], a[:15], basefmt=" ")
    axes[0, 1].set(
        title="The DFT recovers their amplitudes",
        xlabel="Frequency (Hz)",
        ylabel="Amplitude",
        ylim=(0, 2.4),
    )
    shifted = np.roll(x, 16)
    axes[1, 1].plot(t, x, alpha=0.5, label="Original")
    axes[1, 1].plot(t, shifted, color=ORANGE, label="Shifted by 0.125 s")
    axes[1, 1].set(
        title="Same magnitudes; different phases",
        xlabel="Time (seconds)",
        ylabel="Value",
    )
    axes[1, 1].legend(fontsize=9)
    save(fig, "ingredients")
    X = np.fft.fft(x)
    np.testing.assert_allclose(np.abs(np.fft.fft(shifted)), np.abs(X), atol=1e-12)
    np.testing.assert_allclose(np.fft.ifft(X).real, x, atol=1e-12)
    np.testing.assert_allclose(np.var(x), np.sum(np.abs(X[1:]) ** 2) / fs**2)
    np.testing.assert_allclose([a[3], a[8]], [2, 0.75])
    np.testing.assert_allclose(np.var(x), 2.28125)
    print("Two-tone variance:", np.var(x), "amplitudes:", a[3], a[8])

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 4.5), layout="constrained")
    frames = []
    dense = np.linspace(0, 1, 400)
    first = 2 * np.sin(2 * np.pi * 3 * dense)
    second = 0.75 * np.cos(2 * np.pi * 8 * dense)
    for i in range(24):
        for ax in axes:
            ax.clear()
        weight = min(i / 16, 1)
        axes[0].plot(dense, first, label="3 Hz sine")
        axes[0].plot(dense, weight * second, color=ORANGE, label="8 Hz cosine")
        axes[0].legend(loc="upper right", fontsize=9)
        axes[0].set(ylabel="Ingredients", ylim=(-3, 3), xticks=[])
        axes[1].plot(dense, first + weight * second, color=GREEN)
        axes[1].set(ylabel="Sum", xlabel="Time (seconds)", ylim=(-3, 3))
        fig.suptitle(
            f"Add an 8 Hz ingredient: amplitude {0.75 * weight:.2f}", fontsize=14
        )
        frames.append(frame(fig))
    gif(frames, "build-a-wave")
    plt.close(fig)


def spatial_figure():
    n = 128
    y, x = np.mgrid[:n, :n]
    smooth = 0.5 + 0.45 * np.cos(2 * np.pi * 3 * x / n)
    texture = 0.5 + 0.45 * np.cos(2 * np.pi * (18 * x + 10 * y) / n)
    square = np.zeros((n, n))
    square[40:88, 40:88] = 1
    freq = np.fft.fftfreq(n) * n
    fy, fx = np.meshgrid(freq, freq, indexing="ij")
    mask = np.exp(-(fx**2 + fy**2) / (2 * 6**2))
    fig, axes = plt.subplots(3, 3, figsize=(10, 9), layout="constrained")
    for row, (im, name) in enumerate(
        zip(
            [smooth, texture, square],
            [
                "Broad stripes: 3 cycles / width",
                "Fine diagonal texture",
                "Sharp square boundary",
            ],
            strict=True,
        )
    ):
        X = np.fft.fft2(im)
        centered = np.fft.fftshift(np.abs(np.fft.fft2(im - im.mean())))
        db = 20 * np.log10(np.maximum(centered / centered.max(), 1e-4))
        blurred = np.fft.ifft2(X * mask).real
        np.testing.assert_allclose(np.fft.ifft2(X).real, im, atol=1e-12)
        axes[row, 0].imshow(im, cmap="gray", vmin=0, vmax=1, origin="lower")
        axes[row, 0].set(title=name)
        axes[row, 0].axis("off")
        artist = axes[row, 1].imshow(
            db,
            cmap="magma",
            vmin=-80,
            vmax=0,
            origin="lower",
            extent=(-64.5, 63.5, -64.5, 63.5),
        )
        axes[row, 1].set(
            title="Centered spectrum",
            xlabel="Horizontal cycles / width",
            ylabel="Vertical cycles / height",
            xlim=(-32, 32),
            ylim=(-32, 32),
        )
        axes[row, 2].imshow(blurred, cmap="gray", vmin=0, vmax=1, origin="lower")
        axes[row, 2].set(title="Gaussian low-pass")
        axes[row, 2].axis("off")
    fig.colorbar(
        artist,
        ax=axes,
        location="bottom",
        shrink=0.6,
        aspect=40,
        label="dB relative to row's spectral peak",
    )
    save(fig, "spatial-frequency")


def complexity_figure():
    n = 2 ** np.arange(4, 17)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    axes[0].loglog(n, n**2, label=r"Direct DFT: $N^2$")
    axes[0].loglog(n, n * np.log2(n), label=r"FFT: $N\log_2 N$")
    axes[0].set(
        title="Work growth (not benchmark timings)",
        xlabel="Number of samples N",
        ylabel="Work units; constants omitted",
    )
    axes[0].legend(fontsize=9)
    axes[1].loglog(
        n, 16 * n.astype(float) ** 2 / 2**20, label="Explicit complex128 DFT matrix"
    )
    axes[1].loglog(n, 16 * n / 2**20, label="One complex128 array")
    axes[1].set(
        title="Storage for named objects only",
        xlabel="Number of samples N",
        ylabel="MiB",
    )
    axes[1].legend(fontsize=9)
    save(fig, "fft-cost")


def distribution_figure():
    z = np.linspace(-5, 5, 1000)
    omega = np.linspace(-8, 8, 1000)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
    for scale, color in [(0.6, BLUE), (1.2, ORANGE)]:
        density = np.exp(-0.5 * (z / scale) ** 2) / (scale * np.sqrt(2 * np.pi))
        axes[0, 0].plot(z, density, color=color, label=f"sigma = {scale}")
        axes[0, 1].plot(omega, np.exp(-0.5 * scale**2 * omega**2), color=color)
        axes[1, 0].plot(
            z,
            np.where(np.abs(z) <= scale, 1 / (2 * scale), 0),
            color=color,
            label=f"a = {scale}",
        )
        axes[1, 1].plot(omega, np.sinc(scale * omega / np.pi), color=color)
    for row, name in enumerate(["Normal", "Uniform on [-a, a]"]):
        axes[row, 0].set(
            title=f"{name}: probability density", xlabel="Value z", ylabel="Density"
        )
        axes[row, 0].legend(fontsize=9)
        axes[row, 1].axhline(0, color=INK, lw=0.6)
        axes[row, 1].set(
            title="Characteristic function (real here)",
            xlabel="Angular frequency (radians / unit of z)",
            ylabel="phi(omega)",
            ylim=(-0.3, 1.1),
        )
    fig.suptitle("Wider density, narrower central Fourier feature", fontsize=17)
    save(fig, "distributions")
    # Quadrature independently checks the stated analytic transforms.
    grid = np.linspace(-12, 12, 24001)
    normal = np.exp(-(grid**2) / 2) / np.sqrt(2 * np.pi)
    for w in [0, 0.5, 2, 4]:
        numeric = np.trapezoid(normal * np.exp(1j * w * grid), grid)
        np.testing.assert_allclose(numeric, np.exp(-(w**2) / 2), atol=1e-9)
        box = np.linspace(-1, 1, 10001)
        numeric = np.trapezoid(0.5 * np.exp(1j * w * box), box)
        np.testing.assert_allclose(numeric, np.sinc(w / np.pi), atol=2e-8)
    print("Normal and uniform characteristic functions: quadrature checks passed")


def time_frequency_figures():
    fs = 8000
    t = np.arange(2 * fs) / fs
    # Two integer-cycle tones, abrupt switch at 1 second. No recording involved.
    x = np.where(t < 1, np.sin(2 * np.pi * 400 * t), np.sin(2 * np.pi * 1000 * t))
    reverse = x[::-1]
    np.testing.assert_allclose(
        np.abs(np.fft.rfft(x)), np.abs(np.fft.rfft(reverse)), atol=1e-10
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), layout="constrained")
    for row, (signal, name) in enumerate(
        [(x, "400 Hz, then 1000 Hz"), (reverse, "Time-reversed: 1000 Hz, then 400 Hz")]
    ):
        f, a = spectrum(signal, fs)
        axes[row, 0].plot(f, a)
        axes[row, 0].set(
            title=f"Global spectrum: {name}",
            xlabel="Frequency (Hz)",
            ylabel="Amplitude",
            xlim=(0, 1400),
        )
        st = ShortTimeFFT(
            windows.hann(256, sym=False), hop=64, fs=fs, scale_to="magnitude"
        )
        S = np.abs(st.stft(signal)) ** 2
        db = 10 * np.log10(np.maximum(S / S.max(), 1e-6))
        axes[row, 1].imshow(
            db,
            origin="lower",
            aspect="auto",
            extent=st.extent(len(signal)),
            cmap="magma",
            vmin=-60,
            vmax=0,
        )
        axes[row, 1].set(
            title="Spectrogram: timing becomes visible",
            xlabel="Time (seconds)",
            ylabel="Frequency (Hz)",
            ylim=(0, 1400),
            xlim=(0, 2),
        )
    save(fig, "when-tones-happen")

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 4.8), layout="constrained")
    frames = []
    for center in np.linspace(0.16, 1.84, 24):
        for ax in axes:
            ax.clear()
        start = round(center * fs) - 512
        snippet = x[start : start + 1024] * windows.hann(1024, sym=False)
        f = np.fft.rfftfreq(1024, 1 / fs)
        a = 2 * np.abs(np.fft.rfft(snippet)) / windows.hann(1024, sym=False).sum()
        # Plot all samples: decimating for display would alias the 1000 Hz tone.
        axes[0].plot(t, x, lw=0.3, alpha=0.35)
        axes[0].axvspan(start / fs, (start + 1024) / fs, color=ORANGE, alpha=0.35)
        axes[0].set(
            xlim=(0, 2), ylim=(-1.2, 1.2), ylabel="Signal", xlabel="Time (seconds)"
        )
        axes[1].plot(f, a, color=ORANGE)
        axes[1].set(
            xlim=(0, 1400),
            ylim=(0, 1.1),
            ylabel="Local amplitude",
            xlabel="Frequency (Hz)",
        )
        fig.suptitle(f"A 128 ms Hann window, centered at {center:.2f} s", fontsize=14)
        frames.append(frame(fig))
    gif(frames, "sliding-window", duration=150)
    plt.close(fig)

    # Toy voice: additive harmonics, a noise burst, then a rising harmonic stack.
    t = np.arange(3 * fs) / fs
    f0 = np.where(t < 1, 120, 140 + 40 * np.clip(t - 1.4, 0, None))
    phase = 2 * np.pi * np.cumsum(f0) / fs
    voiced = np.zeros_like(t)
    for h in range(1, 25):
        hz = h * f0
        envelope = (
            0.05
            + np.exp(-0.5 * ((hz - 600) / 150) ** 2)
            + 0.6 * np.exp(-0.5 * ((hz - 1500) / 220) ** 2)
        )
        voiced += envelope * np.sin(h * phase) / h
    gate = ((t > 0.1) & (t < 1.0)) | ((t > 1.4) & (t < 2.85))
    noise = np.random.default_rng(11).normal(0, 0.12, len(t))
    voice = voiced * gate + noise * ((t >= 1) & (t < 1.4))
    voice /= np.max(np.abs(voice))
    wavfile.write(ROOT / "synthetic-voice.wav", fs, (voice * 30000).astype(np.int16))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), layout="constrained")
    for ax, length in zip(axes, [256, 1024], strict=True):
        st = ShortTimeFFT(
            windows.hann(length, sym=False), hop=80, fs=fs, scale_to="magnitude"
        )
        power = np.abs(st.stft(voice)) ** 2
        db = 10 * np.log10(np.maximum(power / power.max(), 1e-6))
        art = ax.imshow(
            db,
            origin="lower",
            aspect="auto",
            extent=st.extent(len(voice)),
            cmap="magma",
            vmin=-60,
            vmax=0,
        )
        ax.set(
            xlim=(0, 3),
            ylim=(0, 2500),
            ylabel="Frequency (Hz)",
            xlabel="Time (seconds)",
            title=f"Synthetic voice-like signal: {1000 * length / fs:.0f} ms window; 10 ms hop",
        )
    fig.colorbar(
        art, ax=axes, shrink=0.8, label="Power (dB relative to each panel's peak)"
    )
    save(fig, "voice-windows")
    print("STFT windows: 32/128 ms; grid spacing: 31.25/7.8125 Hz")


if __name__ == "__main__":
    variance_figure()
    ingredients_figure()
    spatial_figure()
    complexity_figure()
    distribution_figure()
    time_frequency_figures()
    for path in sorted(OUT.iterdir()):
        print(f"{path.name}: {path.stat().st_size / 1024:.0f} KiB")
