# # App/models/startup_ensemble.py
# import torch
# import torch.nn as nn

# # =========================
# # Spec CNN Branch
# # =========================
# class SpecCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.AdaptiveAvgPool2d((4, 8))   # 학습 때와 동일해야 함
#         )
#         self.fc = nn.Linear(64 * 4 * 8, 128)

#     def forward(self, x):
#         # x : (B, 1, 64, 128) 같은 스펙트로그램
#         x = self.net(x)
#         x = x.view(x.size(0), -1)
#         return self.fc(x)


# # =========================
# # FFT Branch
# # =========================
# class FFTBranch(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(256, 256), nn.ReLU(),
#             nn.Linear(256, 128)
#         )

#     def forward(self, x):
#         # x : (B, 256)
#         return self.net(x)


# # =========================
# # Energy Branch
# # =========================
# class EnergyBranch(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(19, 64), nn.ReLU(),
#             nn.Linear(64, 64)
#         )

#     def forward(self, x):
#         # x : (B, 19)
#         return self.net(x)


# # =========================
# # Ensemble Model
# # =========================
# class StartupEnsemble(nn.Module):
#     def __init__(self, num_classes: int):
#         super().__init__()
#         self.spec = SpecCNN()
#         self.fft = FFTBranch()
#         self.energy = EnergyBranch()

#         # fc 이름/구성은 학습 때랑 완전히 동일해야 state_dict 로딩이 맞음
#         self.fc = nn.Sequential(
#             nn.Linear(128 + 128 + 64, 128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, num_classes)
#         )

#     def forward(self, spec, fft, energy):
#         """
#         spec   : (B, 1, 64, 128)
#         fft    : (B, 256)
#         energy : (B, 19)
#         """
#         x1 = self.spec(spec)
#         x2 = self.fft(fft)
#         x3 = self.energy(energy)
#         x = torch.cat([x1, x2, x3], dim=1)
#         return self.fc(x)
