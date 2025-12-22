# App/models/idle_ensemble.py

import torch
import torch.nn as nn
import torch.nn.functional as F

spectrogram_input = None
# =============== Waveform CNN ===============
class WaveformCNN1D(nn.Module):
    """Waveform을 입력으로 받는 1D CNN 모델 (예시 구조)"""
    def __init__(
        self,
        num_classes: int,
        input_length: int = 16000,
        base_channels: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(dropout / 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(dropout / 2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(dropout / 2)
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Sequential(
            nn.Linear(base_channels * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x):
        # x : (B, 1, T)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x)          # (B, C, 1)
        x = x.view(x.size(0), -1)        # (B, C)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc_out(x)
        return x


# =============== Masked CNN (Mel) ===============
class MaskedSpatialAttention(nn.Module):
    def __init__(self, importance_mask: torch.Tensor, learnable: bool = True):
        super().__init__()
        self.importance_mask = nn.Parameter(importance_mask, requires_grad=learnable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T)
        mask = self.importance_mask.expand(x.size(0), -1, -1)  # (B, F, T)
        mask = mask.unsqueeze(1)                               # (B, 1, F, T)
        return x * (1 + mask)


class MaskedCNN(nn.Module):
    def __init__(self, num_classes: int,
                 importance_mask: torch.Tensor,
                 in_channels: int = 1,
                 base_channels: int = 32,
                 dropout: float = 0.3):
        super().__init__()

        self.masked_attention = MaskedSpatialAttention(importance_mask, learnable=True)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Sequential(
            nn.Linear(base_channels * 8, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x):
        # x : (B, 1, F, T)
        x = self.masked_attention(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_pool(x)          # (B, C, 1, 1)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc_out(x)
        return x


# =============== Ensemble ===============
class EnsembleVoteModel(nn.Module):
    def __init__(self,
                 waveform_model: nn.Module,
                 spectrogram_model: nn.Module,
                 vote_method: str = "soft"):
        super().__init__()
        self.waveform_model = waveform_model
        self.spectrogram_model = spectrogram_model
        self.vote_method = vote_method

    def forward(self, waveform_input, spectrograam_input):
        # waveform_input : (B, 1, T)
        # spectrogram_input : (B, 1, F, T)
        wf_out = self.waveform_model(waveform_input)
        sp_out = self.spectrogram_model(spectrogram_input)

        if self.vote_method == "hard":
            wf_pred = wf_out.argmax(dim=1)
            sp_pred = sp_out.argmax(dim=1)
            stacked = torch.stack([wf_pred, sp_pred], dim=1)      # (B, 2)
            ensemble_pred = torch.mode(stacked, dim=1)[0]         # (B,)
            logits = F.one_hot(ensemble_pred,
                               num_classes=wf_out.size(1)).float()
            logits = logits * 10.0 - 5.0
            return logits
        else:
            # soft voting
            return (wf_out + sp_out) / 2.0
