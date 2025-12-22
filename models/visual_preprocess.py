# App/models/visual_preprocess.py

import os
import tempfile
from typing import Dict, Any

import numpy as np
import librosa

# -------------------------------------------------
# 0. 멜 스펙트로그램 계산 (네가 쓰던 함수 그대로)
# -------------------------------------------------
def compute_mel_spectrogram(
    filepath,
    sr=None,          # None이면 원래 샘플레이트 유지
    n_fft=2048,
    hop_length=512,
    n_mels=128
):
    """
    Jupyter에서 쓰던 것과 동일한 멜 스펙트로그램 계산 함수.
    반환: (S_db, sr)
      - S_db: (n_mels, time) 형태의 dB 스펙트로그램
      - sr  : 실제 샘플레이트
    """
    y, sr = librosa.load(filepath, sr=sr)
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db, sr, y


# -------------------------------------------------
# 1. 파일 경로 기준 시각화 feature 계산
# -------------------------------------------------
def compute_visual_from_path(filepath: str) -> Dict[str, Any]:
    """
    업로드된 .wav "전체 길이"에 대해
    - Mel spectrogram (dB)
    - MFCC (Mel dB 기반)
    - FFT magnitude (log)
    를 계산해서 웹에서 바로 그릴 수 있게 dict로 반환.

    반환 형식:
        {
            "mel_img": 2D list [n_mels, T_full],
            "mfcc_img": 2D list [n_mfcc, T_full],
            "fft": 1D list [n_fft_bins],
        }
    """
    # 1) 멜 스펙트로그램 (네가 쓰던 방식 그대로)
    mel_db, sr, y = compute_mel_spectrogram(
        filepath,
        sr=None,          # 원래 샘플레이트 그대로
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )  # mel_db: (128, T_full)

    # 2) MFCC (멜 dB에서 바로 계산)
    n_mfcc = 20
    mfcc = librosa.feature.mfcc(
        S=mel_db,
        sr=sr,
        n_mfcc=n_mfcc
    )  # (20, T_full)

    # 3) FFT magnitude (waveform 전체 기준)
    fft_mag = np.abs(np.fft.rfft(y))
    # 너무 길면 앞쪽 2048bin 정도만 사용 (원하면 더 늘려도 됨)
    max_fft_bins = 2048
    if fft_mag.shape[0] > max_fft_bins:
        fft_mag = fft_mag[:max_fft_bins]
    fft_log = np.log1p(fft_mag).astype(float)  # log 스케일

    # 4) JSON 직렬화를 위해 list로 변환
    mel_img = mel_db.astype(float).tolist()
    mfcc_img = mfcc.astype(float).tolist()
    fft_list = fft_log.tolist()

    return {
        "mel_img": mel_img,
        "mfcc_img": mfcc_img,
        "fft": fft_list,
    }


# -------------------------------------------------
# 2. Flask FileStorage 대응 (웹에서 쓰는 입구)
# -------------------------------------------------
def compute_visual_from_filestorage(file_storage) -> Dict[str, Any]:
    """
    Flask의 FileStorage 객체로부터 바로 시각화 feature 계산.
    다른 코드에서 동일 file_storage를 재사용할 수 있도록
    stream 위치를 원래대로 돌려 준다.
    """
    # 스트림 처음으로 이동
    if hasattr(file_storage, "stream") and hasattr(file_storage.stream, "seek"):
        file_storage.stream.seek(0)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp_name = tmp.name
        file_storage.save(tmp_name)

    # 다시 처음 위치로 복구 (다른 처리에도 사용 가능하도록)
    if hasattr(file_storage, "stream") and hasattr(file_storage.stream, "seek"):
        file_storage.stream.seek(0)

    try:
        return compute_visual_from_path(tmp_name)
    finally:
        try:
            os.remove(tmp_name)
        except OSError:
            pass
