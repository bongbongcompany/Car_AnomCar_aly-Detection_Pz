# App/models/md_inference.py

import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import joblib
import pickle
from scipy.interpolate import interp1d
from .visual_preprocess import compute_visual_from_filestorage
from .md_plots import generate_all_plots_from_wav

# -------------------------------------------------------------------
# 공통 설정
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IDLE_F1_SCORE = 0.98      # Idle 엔진 상태 모델의 전체 F1 (예시 값)
STARTUP_F1_SCORE = 0.98   # Startup 엔진 상태 모델의 전체 F1 (예시 값)

# ===================================================================
# 1) BRAKING (Hybrid CNN + FFT 모델)
# ===================================================================

# --- 경로들: 니 프로젝트 구조에 맞게 수정해줘! ---
BRAKE_MODEL_PATH = BASE_DIR / "brakes.pth"


BRAKE_SR = 16000
BRAKE_N_MELS = 130
BRAKE_N_FFT = 2048
BRAKE_HOP_LENGTH = 512
BRAKE_TIME_DURATION = 4          # 4초
BRAKE_INPUT_MEL_SHAPE = (130, 128)
BRAKE_INPUT_FFT_SIZE = 1024
BRAKE_NUM_CLASSES = 2  # normal / abnormal

# 전역 캐시
_BRAKE_MODEL = None
_BRAKE_SCALER_MEL = None
_BRAKE_SCALER_FFT = None


class CNN_Branch(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.drop1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.drop2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(4, 4))
        self.drop3 = nn.Dropout(0.3)

        # 64 * 8 * 8 = 4096 이라는 가정 (130→65→32→8, 128→64→32→8)
        self.fc1 = nn.Linear(4096, 32)
        self.drop4 = nn.Dropout(0.4)

    def forward(self, x):
        # (B, T, M, 1) -> (B, 1, T, M)
        x = x.permute(0, 3, 1, 2)
        x = self.drop1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = self.drop3(self.pool3(F.relu(self.bn3(self.conv3(x)))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop4(x)
        return x


class MLP_Branch(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.drop1(self.bn1(F.relu(self.fc1(x))))
        x = self.drop2(self.bn2(F.relu(self.fc2(x))))
        return x


class Hybrid_Model(nn.Module):
    """ braking 상태에서 사용한 Mel + FFT 융합 모델 """
    def __init__(self, mel_shape, fft_size):
        super().__init__()
        self.cnn_branch = CNN_Branch(mel_shape)
        self.mlp_branch = MLP_Branch(fft_size)

        combined_size = 32 + 64
        self.dense1 = nn.Linear(combined_size, 16)
        self.bn_comb = nn.BatchNorm1d(16)
        self.drop_comb = nn.Dropout(0.2)
        self.output_layer = nn.Linear(16, BRAKE_NUM_CLASSES)

    def forward(self, mel_input, fft_input):
        cnn_features = self.cnn_branch(mel_input)
        mlp_features = self.mlp_branch(fft_input)
        combined = torch.cat((cnn_features, mlp_features), dim=1)
        z = F.relu(self.bn_comb(self.dense1(combined)))
        z = self.drop_comb(z)
        return self.output_layer(z)


def _ensure_brake_artifacts():
    """ braking 모델 + scaler 1회 로드 """
    global _BRAKE_MODEL, _BRAKE_SCALER_MEL, _BRAKE_SCALER_FFT
    if _BRAKE_MODEL is not None:
        return


    # 모델
    model = Hybrid_Model(BRAKE_INPUT_MEL_SHAPE, BRAKE_INPUT_FFT_SIZE).to(DEVICE)
    state = torch.load(BRAKE_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    _BRAKE_MODEL = model


def _brake_preprocess_audio(audio_path: str):
    """
    braking용 전처리: audio_path -> (mel_tensor, fft_tensor)
    스케일러 파일 없이도 동작하도록, mel/fft를 간단한 z-score 정규화만 적용한다.
    """
    y, sr = librosa.load(audio_path, sr=BRAKE_SR, duration=BRAKE_TIME_DURATION)

    # 길이 4초로 맞추기
    target_len = BRAKE_SR * BRAKE_TIME_DURATION
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), "constant")
    else:
        y = y[:target_len]

    # ----------------- Mel 스펙트로그램 -----------------
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=BRAKE_SR,
        n_fft=BRAKE_N_FFT,
        hop_length=BRAKE_HOP_LENGTH,
        n_mels=BRAKE_N_MELS,
    )
    mel = librosa.power_to_db(mel, ref=np.max)  # (n_mels, time)
    mel = mel.T                                  # (time, n_mels)

    # 🔸 CNN 브랜치가 항상 64*8*8=4096 feature를 만들 수 있게 time 프레임 수 128로 고정
    target_frames = BRAKE_INPUT_MEL_SHAPE[1]     # 보통 128
    if mel.shape[0] < target_frames:
        pad = target_frames - mel.shape[0]
        mel = np.pad(mel, ((0, pad), (0, 0)), mode="constant")
    else:
        mel = mel[:target_frames, :]

    # 🔸 z-score 정규화 (학습 때 스케일러 대신)
    mel_mean = mel.mean()
    mel_std = mel.std() + 1e-8
    mel_norm = (mel - mel_mean) / mel_std

    # (B, T, M, C) -> (1, T, M, 1)
    mel_scaled = mel_norm.reshape(1, mel.shape[0], mel.shape[1], 1)

    # ----------------- FFT 특징 -----------------
    stft = np.abs(librosa.stft(y, n_fft=BRAKE_N_FFT, hop_length=BRAKE_HOP_LENGTH))
    fft_feat = stft.mean(axis=1)

    # 길이를 BRAKE_INPUT_FFT_SIZE(지금 1024)로 강제
    if fft_feat.shape[0] < BRAKE_INPUT_FFT_SIZE:
        fft_feat = np.pad(
            fft_feat,
            (0, BRAKE_INPUT_FFT_SIZE - fft_feat.shape[0]),
            mode="constant",
        )
    else:
        fft_feat = fft_feat[:BRAKE_INPUT_FFT_SIZE]

    fft_mean = fft_feat.mean()
    fft_std = fft_feat.std() + 1e-8
    fft_norm = (fft_feat - fft_mean) / fft_std
    fft_feat_scaled = fft_norm.reshape(1, -1)    # (1, 1024)

    mel_tensor = torch.tensor(mel_scaled, dtype=torch.float32)
    fft_tensor = torch.tensor(fft_feat_scaled, dtype=torch.float32)
    return mel_tensor, fft_tensor



def _brake_predict_from_path(audio_path: str):
    _ensure_brake_artifacts()

    mel_tensor, fft_tensor = _brake_preprocess_audio(audio_path)
    mel_tensor = mel_tensor.to(DEVICE)
    fft_tensor = fft_tensor.to(DEVICE)

    with torch.no_grad():
        outputs = _BRAKE_MODEL(mel_tensor, fft_tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))

    label = "정상 (Normal)" if idx == 0 else "문제 발생 (Abnormal)"
    return {
        "predicted_index": idx,
        "predicted_label": label,
        "probabilities": {
            "normal_prob": float(probs[0]),
            "abnormal_prob": float(probs[1]),
        },
    }


def _brake_predict_from_filestorage(file_storage):
    # FileStorage -> temp wav
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        file_storage.save(tmp.name)
        tmp_path = tmp.name
    try:
        return _brake_predict_from_path(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ===================================================================
# 2) STARTUP (Spec + FFT + Energy Ensemble)
# ===================================================================

STARTUP_MODEL_PATH = BASE_DIR / "startup.pth"           # 수정 가능
STARTUP_LABEL_PATH = BASE_DIR / "startup_label_encoder.pkl"
STARTUP_SR = 16000

_STARTUP_MODEL = None
_STARTUP_ENCODER = None
_STARTUP_CLASSES = None


# ---- Startup 모델 아키텍처 (학습 코드와 동일해야 함) ----
class SpecCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 8))
        )
        self.fc = nn.Linear(64 * 4 * 8, 128)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class FFTBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128),
        )

    def forward(self, x):
        return self.net(x)


class EnergyBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(19, 64), nn.ReLU(),
            nn.Linear(64, 64),
        )

    def forward(self, x):
        return self.net(x)


class StartupEnsemble(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.spec = SpecCNN()
        self.fft = FFTBranch()
        self.energy = EnergyBranch()
        self.fc = nn.Sequential(
            nn.Linear(128 + 128 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, spec, fft, energy):
        x1 = self.spec(spec)
        x2 = self.fft(fft)
        x3 = self.energy(energy)
        x = torch.cat([x1, x2, x3], dim=1)
        return self.fc(x)


def _startup_get_melspec(sig, sr=STARTUP_SR, n_mels=64, n_fft=2048,
                         hop_length=256, fix_length=128):
    """
    학습할 때 사용한 것과 최대한 비슷하게 mel-spec 생성 (64, 128)
    """
    S = librosa.feature.melspectrogram(
        y=sig,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    S_db = librosa.power_to_db(S, ref=np.max)  # (n_mels, T)

    # time 길이를 fix_length로 맞추기
    if S_db.shape[1] < fix_length:
        pad_width = fix_length - S_db.shape[1]
        S_db = np.pad(S_db, ((0, 0), (0, pad_width)), mode="constant",
                      constant_values=S_db.min())
    else:
        S_db = S_db[:, :fix_length]

    return S_db.astype(np.float32)


def _ensure_startup_artifacts():
    global _STARTUP_MODEL, _STARTUP_ENCODER, _STARTUP_CLASSES
    if _STARTUP_MODEL is not None:
        return

    # Label Encoder
    _STARTUP_ENCODER = joblib.load(STARTUP_LABEL_PATH)
    _STARTUP_CLASSES = _STARTUP_ENCODER.classes_

    # Model
    model = StartupEnsemble(num_classes=len(_STARTUP_CLASSES)).to(DEVICE)
    state = torch.load(STARTUP_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    _STARTUP_MODEL = model

def _startup_extract_features_from_path(wav_path: str):
    sig, _ = librosa.load(wav_path, sr=STARTUP_SR, mono=True)

    sig = sig / (np.max(np.abs(sig)) + 1e-8)

    if len(sig) >= STARTUP_SR:
        sig = sig[:STARTUP_SR]
    else:
        sig = np.pad(sig, (0, STARTUP_SR - len(sig)))

    # 1) spec (64, 128)
    spec = _startup_get_melspec(sig, sr=STARTUP_SR, fix_length=128)

    # 2) FFT (256)
    fft = np.abs(np.fft.rfft(sig, n=1024))[:256].astype(np.float32)

    # 3) energy (19)
    frames = librosa.util.frame(sig, frame_length=512, hop_length=256)
    energy = np.sum(frames ** 2, axis=0)

    energy_stats = np.array([
        energy.mean(),
        energy.std(),
        energy.max(),
        np.percentile(energy, 75),
        np.percentile(energy, 25),
    ], dtype=np.float32)

    if len(energy) >= 14:
        energy_seq = energy[:14]
    else:
        energy_seq = np.pad(energy, (0, 14 - len(energy)))
    energy_seq = energy_seq.astype(np.float32)

    energy_feat = np.concatenate([energy_stats, energy_seq])  # (19,)

    return spec, fft, energy_feat


def _make_startup_visual_from_path(audio_path: str):
    spec, _, _ = _startup_extract_features_from_path(audio_path)
    return {"spec": spec.tolist()}



def _startup_predict_from_path(wav_path: str):
    _ensure_startup_artifacts()

    spec, fft, energy = _startup_extract_features_from_path(wav_path)

    spec_t = torch.tensor(spec).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    fft_t = torch.tensor(fft).float().unsqueeze(0).to(DEVICE)
    energy_t = torch.tensor(energy).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = _STARTUP_MODEL(spec_t, fft_t, energy_t)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()

    prob_dict = {
        cls: float(p) for cls, p in zip(_STARTUP_CLASSES, probs)
    }
    return prob_dict


def _startup_predict_from_filestorage(file_storage):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        file_storage.save(tmp.name)
        tmp_path = tmp.name
    try:
        return _startup_predict_from_path(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ===================================================================
# 3) IDLE (Waveform + Mel + MFCC EnsembleVoteModel)
# ===================================================================

IDLE_MODEL_PATH = BASE_DIR / "idle.pth"          # 수정 필요
IDLE_IMPORTANCE_MASK_PATH = BASE_DIR / "importance_mask.npy"    # 있으면 사용

# 학습 때 사용한 인덱스 → 레이블 매핑
IDLE_IDX_TO_LABEL = {
    0: 'low_oil',
    1: 'normal_engine_idle',
    2: 'power_steering',
    3: 'serpentine_belt',
}

_IDLE_MODEL = None
_IDLE_IMPORTANCE_MASK = None


class WaveformCNN1D(nn.Module):
    def __init__(self, num_classes: int, input_length: int = 110250,
                 base_channels: int = 32, dropout: float = 0.5):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout / 2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout / 2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout / 2),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Sequential(
            nn.Linear(base_channels * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
            x = x * (1 + mask)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class CRNNMelSpectrogram(nn.Module):
    def __init__(
        self,
        num_classes: int,
        n_mels: int = 128,
        time_frames: int = 87,
        base_channels: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.4,
        rnn_type: str = "LSTM",
        bidirectional: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.n_mels = n_mels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout2d(dropout / 3),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout2d(dropout / 3),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout2d(dropout / 3),
        )

        self.cnn_output_features = base_channels * 4
        self.cnn_output_time = time_frames // 8
        self.cnn_output_freq = n_mels // 8

        rnn_input_size = self.cnn_output_features * self.cnn_output_freq

        if rnn_type.upper() == "LSTM":
            self.rnn = nn.LSTM(
                input_size=rnn_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True,
            )
        elif rnn_type.upper() == "GRU":
            self.rnn = nn.GRU(
                input_size=rnn_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True,
            )
        else:
            raise ValueError("rnn_type must be 'LSTM' or 'GRU'")

        rnn_output_size = hidden_size * 2 if bidirectional else hidden_size

        self.fc1 = nn.Sequential(
            nn.Linear(rnn_output_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x, mask=None):
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                if mask.size(0) == 1:
                    mask = mask.unsqueeze(1)
                else:
                    mask = mask.unsqueeze(1)
            if mask.size(0) == 1 and x.size(0) > 1:
                mask = mask.expand(x.size(0), -1, -1, -1)
            x = x * (1 + mask)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        batch_size = x.size(0)
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(batch_size, x.size(1), -1)

        rnn_out, _ = self.rnn(x)
        rnn_out = rnn_out[:, -1, :]
        x = self.fc1(rnn_out)
        return x


class MFCCCCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        num_mfcc_coeffs: int = 20,
        time_frames: int = 87,
        base_channels: int = 64,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.output_dim = 256
        self.conv1 = self._conv_block(1, base_channels, dropout / 3)
        self.conv2 = self._conv_block(base_channels, base_channels * 2, dropout / 3)
        self.conv3 = self._conv_block(base_channels * 2, base_channels * 4, dropout / 3)

        final_dim = self._calculate_final_dim(base_channels, num_mfcc_coeffs, time_frames)
        self.fc1 = nn.Sequential(
            nn.Linear(final_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def _conv_block(self, in_c, out_c, dropout):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
        )

    def _calculate_final_dim(self, base_channels, num_mfcc_coeffs, time_frames):
        try:
            with torch.no_grad():
                dummy = torch.randn(1, 1, num_mfcc_coeffs, time_frames)
                x = self.conv1(dummy)
                x = self.conv2(x)
                x = self.conv3(x)
                x = F.adaptive_avg_pool2d(x, 1)
                final_dim = x.numel()
                return final_dim if final_dim > 0 else base_channels * 4 * 2 * 6
        except Exception:
            return base_channels * 4 * 2 * 6

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class EnsembleVoteModel(nn.Module):
    def __init__(self, num_classes: int = 4, base_channels_wf: int = 32, base_channels_mel: int = 64, base_channels_mfcc: int = 64, num_mfcc_coeffs: int = 20, time_frames: int = 87):
        super(EnsembleVoteModel, self).__init__()
        
        wf_out_dim = 256
        mel_out_dim = 256
        mfcc_out_dim = 256
        
        self.wf_model = WaveformCNN1D(num_classes=wf_out_dim, base_channels=base_channels_wf) 
        # 🌟 CRNN 모델 사용 (기존 MaskedCNN 대신)
        self.mel_model = CRNNMelSpectrogram(
            num_classes=mel_out_dim,
            n_mels=128,
            time_frames=time_frames,
            base_channels=base_channels_mel,
            hidden_size=128,
            num_layers=2,
            dropout=0.4,
            rnn_type='LSTM',
            bidirectional=True
        )
        self.mfcc_model = MFCCCCNN(num_classes=mfcc_out_dim, num_mfcc_coeffs=num_mfcc_coeffs, time_frames=time_frames, base_channels=base_channels_mfcc) 
        
        total_input_dim = wf_out_dim + mel_out_dim + mfcc_out_dim
        self.final_fc = nn.Linear(total_input_dim, num_classes)
        
    def forward(self, wf, mel, mfcc, wf_mask, importance_mask=None): 
        wf_out = self.wf_model(wf, wf_mask)
        mel_out = self.mel_model(mel, importance_mask)
        mfcc_out = self.mfcc_model(mfcc)
        combined = torch.cat((wf_out, mel_out, mfcc_out), dim=1) 
        output = self.final_fc(combined)
        return output


def _idle_extract_mfcc(y, sr, n_fft=2048, hop_length=512, n_mels=128):
    n_mfcc = 20
    mfccs = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        n_mfcc=n_mfcc,
    )
    mfccs_db = librosa.power_to_db(mfccs, ref=np.max)
    return mfccs_db


def _idle_preprocess_audio(audio_path, sample_rate=22050, duration=2.0,
                           n_mels=128, n_fft=2048, hop_length=512):
    y, sr = librosa.load(str(audio_path), sr=sample_rate)

    # trim
    y_trimmed, _ = librosa.effects.trim(y, top_db=60)
    y_norm = librosa.util.normalize(y_trimmed)

    target_len = int(sample_rate * duration)
    if len(y_norm) < target_len:
        y_padded = np.pad(y_norm, (0, target_len - len(y_norm)), mode="constant")
    elif len(y_norm) > target_len:
        y_padded = y_norm[:target_len]
    else:
        y_padded = y_norm

    mel_spec = librosa.feature.melspectrogram(
        y=y_padded,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    target_time_frames = int(np.ceil(target_len / hop_length))
    cur_tf = mel_spec_db.shape[1]
    if cur_tf < target_time_frames:
        pad = np.zeros((mel_spec_db.shape[0], target_time_frames - cur_tf))
        pad.fill(mel_spec_db.min())
        mel_spec_db = np.concatenate([mel_spec_db, pad], axis=1)
    elif cur_tf > target_time_frames:
        mel_spec_db = mel_spec_db[:, :target_time_frames]

    mfcc_spec_db = _idle_extract_mfcc(y_padded, sample_rate, n_fft, hop_length, n_mels)
    cur_tf = mfcc_spec_db.shape[1]
    if cur_tf < target_time_frames:
        pad = np.zeros((mfcc_spec_db.shape[0], target_time_frames - cur_tf))
        pad.fill(mfcc_spec_db.min())
        mfcc_spec_db = np.concatenate([mfcc_spec_db, pad], axis=1)
    elif cur_tf > target_time_frames:
        mfcc_spec_db = mfcc_spec_db[:, :target_time_frames]

    rms = librosa.feature.rms(y=y_padded)[0]
    if rms.max() > rms.min():
        attention = (rms - rms.min()) / (rms.max() - rms.min())
    else:
        attention = np.ones_like(rms) * 0.5

    target_len_wf = len(y_padded)
    if len(attention) != target_len_wf:
        x_old = np.linspace(0, 1, len(attention))
        x_new = np.linspace(0, 1, target_len_wf)
        f = interp1d(x_old, attention, kind="linear")
        attention = f(x_new)

    waveform = torch.FloatTensor(y_padded).unsqueeze(0).unsqueeze(0)  # (1,1,L)
    mel_t = torch.FloatTensor(mel_spec_db).unsqueeze(0).unsqueeze(0)   # (1,1,n_mels,T)
    mfcc_t = torch.FloatTensor(mfcc_spec_db).unsqueeze(0).unsqueeze(0) # (1,1,n_mfcc,T)
    wf_mask = torch.FloatTensor(attention).unsqueeze(0)              # (1, L)
    return waveform, mel_t, mfcc_t, wf_mask

def _make_idle_visual_from_path(audio_path: str):
    """
    idle 전처리 결과를 웹에서 그리기 좋게 줄인 형태로 반환
    - waveform: 1D (downsample)
    - mel     : 2D (mel x time)
    """
    waveform, mel_spec, _, _ = _idle_preprocess_audio(audio_path)

    wf_np = waveform.squeeze().cpu().numpy()
    mel_np = mel_spec.squeeze().cpu().numpy()

    # 파형은 너무 길면 최대 1000 포인트로 다운샘플
    max_points = 1000
    if wf_np.ndim == 0:
        wf_ds = np.array([float(wf_np)])
    else:
        n = wf_np.shape[0]
        if n > max_points:
            idx = np.linspace(0, n - 1, max_points).astype(int)
            wf_ds = wf_np[idx]
        else:
            wf_ds = wf_np

    return {
        "waveform": wf_ds.tolist(),
        "mel": mel_np.tolist(),
    }

def _ensure_idle_model():
    global _IDLE_MODEL, _IDLE_IMPORTANCE_MASK
    if _IDLE_MODEL is not None:
        return

    model = EnsembleVoteModel(
        num_classes=len(IDLE_IDX_TO_LABEL),
        base_channels_wf=32,
        base_channels_mel=64,
        base_channels_mfcc=64,
    )
    state = torch.load(IDLE_MODEL_PATH, map_location=DEVICE)

    # 🔸 strict=False 로 로드해서 rnn 관련 없는 키는 무시
    incompatible = model.load_state_dict(state, strict=False)

    # 디버그용으로 뭐가 없는지 찍어두면 좋음
    if incompatible.missing_keys:
        print("[IDLE] WARNING - missing keys:", incompatible.missing_keys)
    if incompatible.unexpected_keys:
        print("[IDLE] WARNING - unexpected keys:", incompatible.unexpected_keys)

    model.to(DEVICE)
    model.eval()
    _IDLE_MODEL = model

    if IDLE_IMPORTANCE_MASK_PATH.exists():
        mask_np = np.load(IDLE_IMPORTANCE_MASK_PATH)
        _IDLE_IMPORTANCE_MASK = torch.FloatTensor(mask_np).unsqueeze(0).to(DEVICE)
    else:
        _IDLE_IMPORTANCE_MASK = None



def _idle_predict_from_path(audio_path: str):
    _ensure_idle_model()

    waveform, mel_spec, mfcc_spec, wf_mask = _idle_preprocess_audio(audio_path)
    waveform = waveform.to(DEVICE)
    mel_spec = mel_spec.to(DEVICE)
    mfcc_spec = mfcc_spec.to(DEVICE)
    wf_mask = wf_mask.to(DEVICE)

    with torch.no_grad():
        outputs = _IDLE_MODEL(
            wf=waveform,
            mel=mel_spec,
            mfcc=mfcc_spec,
            wf_mask=wf_mask,
            importance_mask=_IDLE_IMPORTANCE_MASK,
        )
        probs = F.softmax(outputs, dim=1)
        probs_np = probs.cpu().numpy()[0]
        idx = int(np.argmax(probs_np))

    label = IDLE_IDX_TO_LABEL[idx]
    prob_dict = {IDLE_IDX_TO_LABEL[i]: float(p) for i, p in enumerate(probs_np)}
    return label, prob_dict


def _idle_predict_from_filestorage(file_storage):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        file_storage.save(tmp.name)
        tmp_path = tmp.name
    try:
        return _idle_predict_from_path(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
def _make_startup_visual_from_path(audio_path: str):
    """
    startup 전처리 결과 중 mel-spectrogram만 시각화용으로 사용
    """
    spec, _, _ = _startup_extract_features_from_path(audio_path)
    return {
        "spec": spec.tolist(),
    }

import tempfile
import os
from typing import Optional, Dict, Any

from .md_plots import generate_all_plots_from_wav

# ===================================================================
# 최종 공용 엔트리포인트
# ===================================================================
from typing import Dict, Any, Optional
import tempfile
import os

from .md_plots import generate_all_plots_from_wav   # ← 상단 import에 이미 있다면 중복 제거


def run_md_inference(models, state, file_storage) -> Dict[str, Any]:
    """
    공통 엔트리포인트.
    models 인자는 사용하지 않고, 이 파일 내부의
    _brake_predict_from_path / _idle_predict_from_path /
    _startup_predict_from_path 를 사용한다.

    반환 형식:
        {
            "state": "idle",
            "label": "...",
            "score": 0.68,
            "probs": {...},         # 각 클래스 확률
            "images": {
                "waveform": "/static/md_plots/....png",
                "mel": "...",
                "fft": "...",
                "f1": "...",
            },
            "f1_scores": {...}      # F1 바 차트용 (여기선 probs 그대로 사용)
        }
    """
    state = (state or "").lower()

    # 1) 업로드 파일을 임시 wav 로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        file_storage.save(tmp.name)
        wav_path = tmp.name

    try:
        f1_dict: Optional[Dict[str, float]] = None

        # =====================================================
        # 2) 상태별 모델 추론
        # =====================================================
        if state == "braking":
            # path 버전 사용
            pred = _brake_predict_from_path(wav_path)

            label = pred["predicted_label"]
            probs_raw = pred["probabilities"]   # {"normal_prob": ..., "abnormal_prob": ...}
            # 확률 중 가장 큰 값 = score
            score = float(max(probs_raw.values()))

            # F1 그래프용: 보기 좋은 이름으로 다시 매핑 (필요 없으면 그대로 써도 됨)
            f1_dict = {
                "brake_normal": float(probs_raw.get("normal_prob", 0.0)),
                "brake_abnormal": float(probs_raw.get("abnormal_prob", 0.0)),
            }

            base_result = {
                "state": "braking",
                "label": label,
                "score": score,
                "probs": probs_raw,   # 원본 확률 dict
            }

        elif state == "idle":
            # (label, probs_dict) 반환
            label, probs = _idle_predict_from_path(wav_path)
            # probs: {"low_oil": ..., "normal_engine_idle": ..., "power_steering": ..., "serpentine_belt": ...}
            score = float(max(probs.values()))

            # idle 은 4개 클래스 확률을 그대로 F1 막대용으로 사용
            f1_dict = {k: float(v) for k, v in probs.items()}

            base_result = {
                "state": "idle",
                "label": label,
                "score": score,
                "probs": probs,
            }

        elif state == "startup":
            # probs: {"class0": p0, "class1": p1, ...}
            probs = _startup_predict_from_path(wav_path)
            label = max(probs, key=probs.get)
            score = float(probs[label])

            # startup 은 클래스 개수(3개)를 그대로 막대 개수로 사용
            f1_dict = {k: float(v) for k, v in probs.items()}

            base_result = {
                "state": "startup",
                "label": label,
                "score": score,
                "probs": probs,
            }

        else:
            raise ValueError(f"Unknown state: {state}")

        # =====================================================
        # 3) PNG 이미지 생성 (wave / mel / fft / f1)
        # =====================================================
        images = generate_all_plots_from_wav(wav_path, f1_dict=f1_dict)

        base_result["images"] = images
        if f1_dict is not None:
            base_result["f1_scores"] = f1_dict

        return base_result

    finally:
        # 임시 wav 삭제
        try:
            os.remove(wav_path)
        except OSError:
            pass
