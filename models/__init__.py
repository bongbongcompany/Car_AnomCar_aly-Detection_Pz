# App/models/__init__.py

"""
MD 관련 모델 로딩/추론 함수만 모아두는 패키지 초기화 파일.
Flask 앱 생성(create_app)이나 views 관련 코드는 절대 넣지 마세요.
"""

from .model_loader import load_md_models
from .md_inference import run_md_inference  # md_inference.py 에 만들었다면

__all__ = ["load_md_models", "run_md_inference"]
