# App/models/md_plots.py

import os
from pathlib import Path
from uuid import uuid4
from typing import Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Flask 서버에서 PNG만 뽑을 때 필수
import matplotlib.pyplot as plt
import librosa
import librosa.display

# 🔹 전역 스타일: 글씨/축/틱 다 흰색
matplotlib.rcParams.update({
    "text.color": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "axes.edgecolor": "white",
    "axes.linewidth": 0.09,   # ← 기본 축선 두께 (기본 0.8 정도)
})

# models 디렉토리 기준으로 static 하위에 md_plots 폴더 생성
BASE_DIR = Path(__file__).resolve().parents[1]   # .../App
STATIC_DIR = BASE_DIR / "static"
PLOT_DIR = STATIC_DIR / "md_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def _new_filename(kind: str) -> str:
    """md_plots 폴더 안에 저장할 랜덤 파일명 생성."""
    return f"md_{uuid4().hex}_{kind}.png"


# =========================================================
# Waveform
# =========================================================
def plot_waveform_png(y: np.ndarray, sr: int) -> str:
    fname = _new_filename("waveform")
    out_path = PLOT_DIR / fname

    fig, ax = plt.subplots(figsize=(6, 2), dpi=150)
    # 🔹 배경 투명
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    librosa.display.waveshow(y, sr=sr,  linewidth=0.01, color="#FF7B00", ax=ax)
    ax.axis("off")  # 글씨 필요 없으면 축 숨김

    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)

    return "/static/md_plots/" + fname


# =========================================================
# Mel Spectrogram
# =========================================================
def plot_mel_png(S_db: np.ndarray, sr: int, hop_length: int = 512) -> str:
    fname = _new_filename("mel")
    out_path = PLOT_DIR / fname

    fig, ax = plt.subplots(figsize=(6, 2.4), dpi=150)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    img = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        cmap="magma",
        ax=ax,
    )

    # 🔽 글씨 조금 더 작게
    ax.set_title("Spectrogram", color="white", fontsize=7, pad=3)  # 9 -> 8
    ax.set_xlabel("Time", fontsize=7)                              # 8 -> 7
    ax.set_ylabel("Hz", fontsize=7)                                # 8 -> 7

    # 눈금도 더 작게
    ax.tick_params(axis="both", labelsize=6, colors="white")       # 7 -> 6
    for spine in ax.spines.values():
        spine.set_color("white")

    # 컬러바도 슬림하게 + 작은 글씨
    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.01, fraction=0.035)
    cbar.ax.set_facecolor("none")
    cbar.outline.set_edgecolor("white")
    cbar.ax.tick_params(labelsize=6, colors="white")               # 7 -> 6

    fig.tight_layout(pad=0.1)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.0, transparent=True)
    plt.close(fig)

    return "/static/md_plots/" + fname


# =========================================================
# FFT
# =========================================================
def plot_fft_png(y: np.ndarray, sr: int) -> str:
    fname = _new_filename("fft")
    out_path = PLOT_DIR / fname

    # FFT 계산 (0~5000Hz)
    fft = np.abs(np.fft.rfft(y))
    freqs = np.linspace(0, sr / 2, len(fft))
    max_freq = 5000.0
    mask = freqs <= max_freq
    freqs = freqs[mask]
    fft = fft[mask]

    # smoothing
    kernel = np.ones(7) / 7.0
    fft_smooth = np.convolve(fft, kernel, mode="same")

    fig, ax = plt.subplots(figsize=(6, 2.4), dpi=150)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    # 🔽 라인 더 얇게 (1.1 -> 0.8)
    ax.plot(freqs, fft_smooth, linewidth=0.3, color="#35cfff")

    # 🔽 라벨/틱 폰트 더 작게
    ax.set_xlabel("Frequency (Hz)", fontsize=7)                # 8 -> 7
    ax.set_ylabel("Amplitude", fontsize=7)                     # 8 -> 7
    ax.tick_params(axis="both", labelsize=6, colors="white")   # 7 -> 6

    # 🔽 그리드 선도 살짝 얇게 (0.5 -> 0.35)
    ax.grid(color="white", alpha=0.08, linewidth=0.3)
    ax.set_xlim(0, max_freq)

    for spine in ax.spines.values():
        spine.set_color("white")

    fig.tight_layout(pad=0.1)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.0, transparent=True)
    plt.close(fig)

    return "/static/md_plots/" + fname

# =========================================================
# F1 bar chart
# =========================================================
def plot_f1_png(f1_dict: Dict[str, float]) -> str:
    fname = _new_filename("f1")
    out_path = PLOT_DIR / fname

    labels = list(f1_dict.keys())
    values = [float(v) for v in f1_dict.values()]

    fig, ax = plt.subplots(figsize=(4.8, 2.2), dpi=150)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    bars = ax.bar(labels, values, color="#2ef5a4")

    ax.set_ylim(0.0, 1.0)
    # 🔽 y 라벨 폰트 더 작게
    ax.set_ylabel("F1 score", fontsize=6)                      # 8 -> 7
    ax.tick_params(axis="x", labelsize=6, rotation=25, colors="white")  # 7 -> 6
    ax.tick_params(axis="y", labelsize=6, colors="white")               # 7 -> 6

    # 위에 작은 퍼센트 표시 (폰트도 조금 더 작게)
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            v + 0.02,
            f"{v*100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=6,  # 7 -> 6
            color="white",
        )

    for spine in ax.spines.values():
        spine.set_color("white")

    fig.tight_layout(pad=0.1)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.0, transparent=True)
    plt.close(fig)

    return "/static/md_plots/" + fname


# =========================================================
# 메인: wav → 모든 PNG 생성
# =========================================================
def generate_all_plots_from_wav(
    wav_path: str,
    f1_dict: Optional[Dict[str, float]] = None
) -> Dict[str, str]:
    """
    WAV 파일 경로 하나로부터
    - waveform
    - mel spectrogram
    - FFT
    - F1 bar (선택)
    의 PNG를 생성하고, 웹에서 쓸 수 있는 URL을 dict로 반환.
    """
    y, sr = librosa.load(wav_path, sr=None, mono=True)

    # 멜 스펙트로그램
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    imgs: Dict[str, str] = {
        "waveform": plot_waveform_png(y, sr),
        "mel": plot_mel_png(S_db, sr, hop_length=512),
        "fft": plot_fft_png(y, sr),
    }

    if f1_dict is not None:
        imgs["f1"] = plot_f1_png(f1_dict)

    return imgs
