# import io
# import torch
# import soundfile as sf
# import numpy as np

# def run_md_inference(models, state, file_storage):
#     """
#     models: create_app에서 넣어준 dict
#         - 'braking', 'idle', 'startup'  : torch 모델
#         - 'startup_encoder'            : LabelEncoder (pkl 로드)
#     state: 'braking' / 'idle' / 'startup'
#     file_storage: werkzeug.datastructures.FileStorage (.wav 파일)
#     """

#     # 1) wav -> numpy
#     wav_bytes = file_storage.read()
#     audio_data, sr = sf.read(io.BytesIO(wav_bytes))

#     # 여기서 원하는 전처리(리샘플링, mono, normalize, STFT 등)를 해줘야 함
#     # 예시는 아주 단순한 형태로만 남겨둠
#     if audio_data.ndim > 1:     # stereo -> mono
#         audio_data = np.mean(audio_data, axis=1)

#     # 예: (1, 1, N) 형태로 만든다고 가정
#     x = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

#     if state == "braking":
#         model = models["braking"]
#         with torch.no_grad():
#             logits = model(x)
#         pred = torch.argmax(logits, dim=1).item()
#         # 예시 텍스트
#         text = f"[제동 상태] 예측 클래스: {pred}"

#     elif state == "idle":
#         model = models["idle"]
#         with torch.no_grad():
#             logits = model(x)
#         pred = torch.argmax(logits, dim=1).item()
#         text = f"[주행 상태] 예측 클래스: {pred}"

#     else:  # startup
#         model = models["startup"]
#         encoder = models["startup_encoder"]  # LabelEncoder
#         with torch.no_grad():
#             logits = model(x)
#         idx = torch.argmax(logits, dim=1).item()
#         label = encoder.inverse_transform([idx])[0]
#         text = f"[시동 상태] 예측 라벨: {label}"

#     return {
#         "state": state,
#         "text": text,
#     }
