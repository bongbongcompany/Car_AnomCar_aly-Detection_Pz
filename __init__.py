# App/__init__.py
from flask import Flask
from pymongo import MongoClient
from .views import main_view, auth_view
from dotenv import load_dotenv
import os
from .models.model_loader import load_md_models   # 🔹 모델 로더 가져오기


# ----- MongoDB 연결 (전역) -----
mongo_client = MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["car_web"]      # DB 이름
users_col = mongo_db["users"]           # users 컬렉션


def create_app():
    app = Flask(__name__)

    # 세션용 secret key (임의 문자열)
    app.secret_key = "1234"

    # Flask app 객체에 DB 핸들 붙이기
    app.mongo_db = mongo_db
    app.users_col = users_col

    # 🔹 MD 모델들을 한 번만 로드해서 config에 저장
    app.config["MD_MODELS"] = load_md_models()

    # 카카오 설정
    app.config['KAKAO_REST_API_KEY'] = os.getenv('KAKAO_REST_API_KEY')
    app.config['KAKAO_CLIENT_SECRET'] = os.getenv('KAKAO_CLIENT_SECRET')
    app.config['KAKAO_REDIRECT_URI'] = os.getenv('KAKAO_REDIRECT_URI')

    # --- Naver ---
    app.config['NAVER_CLIENT_ID'] = os.getenv('NAVER_CLIENT_ID')
    app.config['NAVER_CLIENT_SECRET'] = os.getenv('NAVER_CLIENT_SECRET')
    app.config['NAVER_REDIRECT_URI'] = os.getenv('NAVER_REDIRECT_URI')

    # ★ 문의 메일용 설정 (네이버 예시)
    app.config["MAIL_SERVER"] = "smtp.naver.com"
    app.config["MAIL_PORT"] = 587
    app.config["MAIL_USE_TLS"] = True
    app.config["MAIL_USERNAME"] = os.getenv("MAIL_USERNAME")
    app.config["MAIL_PASSWORD"] = os.getenv("MAIL_PASSWORD")


    # 🔹 블루프린트 등록은 여기서

    app.register_blueprint(main_view.main_bp)
    app.register_blueprint(auth_view.auth_bp, url_prefix="/auth")

    return app
