# App/models/brake_hybrid.py
import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_MEL_SHAPE = (130, 128)
INPUT_FFT_SIZE = 128

class CNN_Branch(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        # Conv Block 1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.drop1 = nn.Dropout(0.2)

        # Conv Block 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.drop2 = nn.Dropout(0.2)

        # Conv Block 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(4, 4))
        self.drop3 = nn.Dropout(0.3)

        # 64 * 8 * 8 = 4096 이라는 가정 (130→65→32→8, 128→64→32→8)
        self.fc1 = nn.Linear(4096, 32)
        self.drop4 = nn.Dropout(0.4)

    def forward(self, x):
        # 학습 때: (B, T, M, C) -> 여기선 (B, 1, T, M)으로 맞춰둔다고 가정
        # 만약 (B, T, M, 1) 형태라면 아래 permute 유지
        x = x.permute(0, 3, 1, 2)  # (B, C, T, M)

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
    """
    braking 상태에서 사용한 Mel + FFT 융합 모델
    """
    def __init__(self, mel_shape, fft_size):
        super().__init__()
        self.cnn_branch = CNN_Branch(mel_shape)
        self.mlp_branch = MLP_Branch(fft_size)

        combined_size = 32 + 64  # CNN 32 + MLP 64

        self.dense1 = nn.Linear(combined_size, 16)
        self.bn_comb = nn.BatchNorm1d(16)
        self.drop_comb = nn.Dropout(0.2)
        self.output_layer = nn.Linear(16, 2)  # 정상/이상 2클래스

    def forward(self, mel_input, fft_input):
        cnn_features = self.cnn_branch(mel_input)
        mlp_features = self.mlp_branch(fft_input)

        combined = torch.cat((cnn_features, mlp_features), dim=1)

        z = F.relu(self.bn_comb(self.dense1(combined)))
        z = self.drop_comb(z)

        return self.output_layer(z)
