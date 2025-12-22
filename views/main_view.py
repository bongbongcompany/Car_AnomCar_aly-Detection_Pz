from flask import Blueprint, render_template, redirect, url_for, session, request ,current_app ,jsonify, flash
from App.models.md_inference import run_md_inference  
main_bp = Blueprint("main", __name__, url_prefix="/")
import time
import smtplib
from email.message import EmailMessage
from datetime import datetime
import secrets 
from bson import ObjectId

@main_bp.route("/", methods=['GET'])
def index():
    # 1. 요청 데이터 읽기
    # 2. 요청 처리
    # 3. 응답 컨텐츠 생산 ( template에 요청 )
     t0 = time.time()
     resp = render_template('index.html')
     print("index render:", time.time() - t0, "sec")

     return resp

# Md → 고장 탐지 페이지
@main_bp.route("/md", methods=["GET"])
def md_page():
    return render_template("md.html")


@main_bp.route("/md/predict", methods=["POST"])
def md_predict():
    # 상태 (braking / idle / startup)
    state = request.form.get("state")
    audio_file = request.files.get("audio")

    if not state or not audio_file:
        return jsonify({"ok": False, "error": "missing_state_or_audio"}), 400

    models = current_app.config.get("MD_MODELS")
    if models is None:
        return jsonify({"ok": False, "error": "models_not_loaded"}), 500

    try:
        result = run_md_inference(models, state, audio_file)
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        print("inference error:", e)  # 터미널에서 에러 확인용
        return jsonify({"ok": False, "error": str(e)}), 500
    
# Em → 실시간 페이지
@main_bp.route("/em", methods=["GET"])
def em_page():
    return render_template("em.html")

# Ev → 마이페이지
@main_bp.route("/ev", methods=["GET"])
def ev_page():
    # 로그인 안 되어 있으면 로그인 페이지로
    if "user_id" not in session:
        return redirect(url_for("auth.login"))

    # 세션에서 기본 정보 꺼내기
    user_id = session.get("user_id")
    user_email = session.get("user_email")
    user_name = session.get("user_name")

    # MongoDB 컬렉션
    db = current_app.mongo_db          # __init__.py 에서 app.mongo_db 해둔 거
    inquiries_col = db["inquiries"]

    # 1) 1:1 문의 총 개수
    inquiry_count = inquiries_col.count_documents({"user_id": user_id})

    # 2) 최근 1:1 문의 5개 (필요하면 10으로 바꿔도 됨)
    my_inquiries = list(
        inquiries_col.find({"user_id": user_id})
                     .sort("created_at", -1)
                     .limit(5)
    )

    # 템플릿에 데이터 넘기기
    return render_template(
        "ev.html",
        user_email=user_email, user_name=user_name, inquiry_count=inquiry_count, my_inquiries=my_inquiries,)

@main_bp.route("/cs", methods=["GET", "POST"])
def cs_page():
    if request.method == "POST":
        # 1. 폼 값 읽기
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        subject = request.form.get("subject", "").strip()
        message = request.form.get("message", "").strip()

        db = current_app.mongo_db
        inquiries_col = db["inquiries"]

        # ★ 관리용 토큰 하나 생성
        admin_token = secrets.token_urlsafe(16)

        # 2. DB에 1:1 문의 저장
        doc = {
            "user_id": session.get("user_id"),
            "user_email": session.get("user_email"),
            "name": name,
            "email": email,
            "subject": subject,
            "message": message,
            "status": "접수",          # 처음에는 '접수'
            "answer": None,
            "created_at": datetime.utcnow(),
            "answered_at": None,
            "admin_token": admin_token,  # ★ 여기 한 줄만 추가된 느낌
        }
        inquiries_col.insert_one(doc)

        # 3. 메일 본문 만들기
        full_subject = f"[1대1 문의] {subject}"

        confirm_url = url_for(
            "main.mark_inquiry_checked",  # 아래에서 만들 라우트 이름
            token=admin_token,
            _external=True,
        )

        full_body = f"""보낸 사람: {name} <{email}>

내용:
{message}

-------------------------

※ 이 문의를 '확인' 상태로 표시하려면 아래 링크를 클릭하세요.
{confirm_url}
"""

        # 4. 메일 전송
        try:
            _send_contact_email(full_subject, full_body)
            flash("문의가 정상적으로 전송되었습니다. 빠른 시일 내에 답변드릴게요 :)", "success")
        except Exception as e:
            print("메일 전송 오류:", e)
            flash("문의 메일 전송 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.", "error")

        return redirect(url_for("main.cs_page"))

    # GET 요청
    return render_template("cs.html")


def _send_contact_email(subject: str, body: str):
    """고객센터 1:1 문의 메일 보내기 (네이버 SMTP 예시)"""
    app = current_app

    smtp_server = app.config["MAIL_SERVER"]
    smtp_port = app.config["MAIL_PORT"]
    username = app.config["MAIL_USERNAME"]
    password = app.config["MAIL_PASSWORD"]
    to_addr = app.config.get("MAIL_TO", username)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = username
    msg["To"] = to_addr
    msg.set_content(body)

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        if app.config.get("MAIL_USE_TLS", False):
            server.starttls()
        server.login(username, password)
        server.send_message(msg)

@main_bp.route("/inquiry/checked/<token>", methods=["GET"])
def mark_inquiry_checked(token):
    """메일 속 링크를 눌렀을 때 해당 문의 status를 '확인'으로 변경"""
    db = current_app.mongo_db
    inquiries_col = db["inquiries"]

    result = inquiries_col.find_one_and_update(
        {"admin_token": token},
        {"$set": {"status": "확인"}}
    )

    if result:
        # 간단한 안내 페이지
        return """
        <html>
          <body style="background:#0b0c1a; color:#fff; font-family:pretendard, sans-serif;">
            <div style="max-width:480px; margin:80px auto; text-align:center;">
              <h2>문의 상태가 '확인'으로 변경되었습니다.</h2>
              <p>이제 마이페이지에서 초록색 '확인' 배지로 표시됩니다.</p>
              <a href="/" style="color:#4ce1c6;">메인으로 돌아가기</a>
            </div>
          </body>
        </html>
        """
    else:
        return "유효하지 않은 링크이거나 이미 처리된 문의입니다.", 404

# map → 로드맵 페이지
@main_bp.route("/map", methods=["GET"])
def map_page():
    return render_template("map.html")


from .crawler_oil import get_oil_prices

@main_bp.route("/oi")
def oi_page():
    oil_data = get_oil_prices()
    oil_items = oil_data.get("items", [])
    return render_template("oi.html", oil_data=oil_data, oil_items=oil_items)
