# App/models/model_loader.py

# from pathlib import Path
# import joblib
# import torch
# import numpy as np
# from . import md_inference
# # md_inference 쪽에 이미 정의된 아키텍처/경로/디바이스 재사용
# from .md_inference import (
#     DEVICE,
#     IDLE_MODEL_PATH,
#     IDLE_IMPORTANCE_MASK_PATH,
#     IDLE_IDX_TO_LABEL,
#    EnsembleVoteModel,
#     StartupEnsemble,
#     STARTUP_MODEL_PATH,
#     STARTUP_LABEL_PATH,
#     # 💡 md_inference.py의 변수 이름을 사용
#     BRAKE_INPUT_MEL_SHAPE,
#     BRAKE_INPUT_FFT_SIZE,
# )


# from .idle_ensemble import WaveformCNN1D, MaskedCNN, EnsembleVoteModel
# from .brake_hybrid import Hybrid_Model

# # 이 파일이 있는 폴더: App/models
# MODEL_DIR = Path(__file__).resolve().parent

# # ----- braking (hybrid) 경로 -----
# BRAKE_MODEL_PATH = MODEL_DIR /"brakes.pth"


# # ==========================================================
# # 1) braking 하이브리드 모델 로더 (Hybrid_Model)
# # ==========================================================
# # ==========================================================
# # 1) braking 하이브리드 모델 로더 (Hybrid_Model)
# # ==========================================================


# def _load_braking_model(device: str = "cpu"):
#     device = torch.device(device)
#     model = Hybrid_Model(
#         mel_shape=BRAKE_INPUT_MEL_SHAPE,
#         fft_size=BRAKE_INPUT_FFT_SIZE,
#     ).to(device)

#     state_dict = torch.load(BRAKE_MODEL_PATH, map_location=device)
#     model.load_state_dict(state_dict)
#     model.eval()
#     print(f"[MD] braking model loaded from {BRAKE_MODEL_PATH}")
#     return model

# def load_md_models():
#     models = {}
#     models["braking"] = _load_braking_model(device=DEVICE)
#     print("MD_MODELS loaded:", {k: type(v) for k, v in models.items()})
#     return models
# # ------------------------------------------------
# # 1) idle 앙상블 모델 로더 (PyTorch)
# # ------------------------------------------------
# def _load_idle_model(device: str = "cpu"):
#     """
#     idle.pth 에 저장된 state_dict를 이용해
#     md_inference와 동일한 EnsembleVoteModel 구조에 가중치를 로드한다.
#     braking은 여기서 다루지 않음 (md_inference 내부에서 처리).
#     """
#     # Flask 서버에서 CPU만 쓸 거라면 DEVICE 대신 새로 잡아도 됨
#     device = torch.device(device)

#     num_classes = len(IDLE_IDX_TO_LABEL)  # idle 클래스 개수 :contentReference[oaicite:3]{index=3}

#     # 중요도 마스크 로드 (있으면)
#     importance_mask = None
#     if IDLE_IMPORTANCE_MASK_PATH.exists():
#         mask_np = np.load(IDLE_IMPORTANCE_MASK_PATH)
#         # md_inference._ensure_idle_model 과 동일한 형태 (1, 128, 216) 참고 :contentReference[oaicite:4]{index=4}
#         importance_mask = torch.FloatTensor(mask_np).unsqueeze(0).to(device)

#     # md_inference와 같은 구조의 앙상블 모델 생성 :contentReference[oaicite:5]{index=5}
#     model = EnsembleVoteModel(
#         num_classes=num_classes,
#         base_channels_wf=32,
#         base_channels_mel=64,
#         base_channels_mfcc=64,
#     ).to(device)

#     state_dict = torch.load(IDLE_MODEL_PATH, map_location=device)
#     model.load_state_dict(state_dict)
#     model.eval()

#     return model, importance_mask


# # ------------------------------------------------
# # 2) startup 앙상블 모델 로더 (PyTorch)
# # ------------------------------------------------
# def _load_startup_model(device: str = "cpu"):
#     """
#     startup.pth + startup_label_encoder.pkl 로부터
#     StartupEnsemble 모델과 LabelEncoder 를 로드한다.
#     """
#     device = torch.device(device)

#     # LabelEncoder 로드 :contentReference[oaicite:6]{index=6}
#     encoder = joblib.load(STARTUP_LABEL_PATH)
#     num_classes = len(encoder.classes_)

#     # md_inference 와 동일한 StartupEnsemble 구조 사용 :contentReference[oaicite:7]{index=7}
#     model = StartupEnsemble(num_classes=num_classes).to(device)

#     state_dict = torch.load(STARTUP_MODEL_PATH, map_location=device)
#     model.load_state_dict(state_dict)
#     model.eval()

#     return model, encoder


# # ==========================================================
# # 4) Flask 에서 사용할 최종 dict 생성
# # ==========================================================
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent

def load_md_models():
    """
    MD 전용 모델은 md_inference.py 에서 lazy-load 되기 때문에
    여기서는 형식 맞추기용 빈 dict만 반환한다.
    """
    models = {}
    print("MD_MODELS loaded (placeholder):", models)
    return models