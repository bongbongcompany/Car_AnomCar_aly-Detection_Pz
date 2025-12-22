# App/views/auth_view.py
from flask import (
    Blueprint, render_template, request,
    redirect, url_for, session, current_app ,jsonify, flash
)
from werkzeug.security import generate_password_hash, check_password_hash
import requests
import secrets

auth_bp = Blueprint("auth", __name__, url_prefix="/")


def get_users_col():
    """현재 Flask app에 붙어있는 users_col 반환."""
    return current_app.users_col

# ---------------- 이메일 중복 확인 ----------------
@auth_bp.route("/check_email")
def check_email():
    users_col = get_users_col()
    email = request.args.get("email", "").strip()

    if not email:
        # 클라이언트에서 에러 메시지 표시
        return jsonify({"ok": False, "error": "no_email"})

    exists = users_col.find_one({"email": email}) is not None
    return jsonify({"ok": True, "exists": exists})

# ---------------- 회원가입 ----------------
@auth_bp.route("/signup", methods=["GET", "POST"])
def signup():
    users_col = get_users_col()  # ★ 여기서 컬렉션 가져옴

    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        name = request.form.get("name", "").strip()

        error = None

        if not email or not password:
            error = "이메일과 비밀번호를 모두 입력해 주세요."
        elif users_col.find_one({"email": email}):
            error = "이미 가입된 이메일입니다."

        if error:
            return render_template("login.html", mode="signup", error=error)

        # 비밀번호 해시 생성
        pw_hash = generate_password_hash(password)

        user_doc = {
            "email": email,
            "password_hash": pw_hash,
            "name": name,
        }
        result = users_col.insert_one(user_doc)

        # 바로 로그인 상태로 만들기
        session["user_id"] = str(result.inserted_id)
        session["user_email"] = email
        session["user_name"] = name

        return redirect(url_for("main.login"))

    # GET 요청
    return render_template("login.html", mode="signup")


# ---------------- 로그인 ----------------
@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    users_col = get_users_col()  # ★ 여기서도 컬렉션 가져옴

    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()

        user = users_col.find_one({"email": email})
        error = None

        if not user:
            error = "해당 이메일로 가입된 계정이 없습니다."
        elif not check_password_hash(user["password_hash"], password):
            error = "비밀번호가 올바르지 않습니다."

        if error:
            return render_template("login.html", mode="login", error=error)

        # 로그인 성공 → 세션에 저장
        session["user_id"] = str(user["_id"])
        session["user_email"] = user["email"]
        session["user_name"] = user.get("name", "")

        return redirect(url_for("main.ev_page"))

    # GET 요청
    return render_template("login.html", mode="login")

# ---------------- 로그아웃 ----------------
@auth_bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("auth.login"))

# ---------------- 카카오 로그인 ----------------
@auth_bp.route("/login/kakao")
def kakao_login():
    """
    카카오 로그인 페이지로 리다이렉트
    """
    kakao_client_id = current_app.config.get("KAKAO_REST_API_KEY")
    redirect_uri = current_app.config.get("KAKAO_REDIRECT_URI")

    if not kakao_client_id or not redirect_uri:
        # 설정이 안 돼 있으면 그냥 에러 띄우고 일반 로그인 페이지로
        flash("카카오 로그인 설정이 없습니다. 관리자에게 문의해 주세요.", "error")
        return redirect(url_for("auth.login"))

    kakao_auth_url = (
        "https://kauth.kakao.com/oauth/authorize"
        f"?response_type=code&client_id={kakao_client_id}&redirect_uri={redirect_uri}"
    )
    return redirect(kakao_auth_url)


@auth_bp.route("/login/kakao/callback")
def kakao_callback():
    """
    카카오에서 code를 받아와서 토큰 발급 → 사용자 정보 조회 → 우리 서비스 로그인 처리
    """
    code = request.args.get("code")
    if not code:
        flash("카카오 로그인 실패: code가 없습니다.", "error")
        return redirect(url_for("auth.login"))

    # 1) 액세스 토큰 요청
    token_url = "https://kauth.kakao.com/oauth/token"
    data = {
        "grant_type": "authorization_code",
        "client_id": current_app.config.get("KAKAO_REST_API_KEY"),
        "redirect_uri": current_app.config.get("KAKAO_REDIRECT_URI"),
        "code": code,
    }
    client_secret = current_app.config.get("KAKAO_CLIENT_SECRET")
    if client_secret:
        data["client_secret"] = client_secret

    try:
        token_res = requests.post(token_url, data=data)
        token_res.raise_for_status()
    except requests.RequestException:
        current_app.logger.exception("카카오 토큰 요청 실패")
        flash("카카오 로그인 중 오류가 발생했습니다. (token)", "error")
        return redirect(url_for("auth.login"))

    token_json = token_res.json()
    access_token = token_json.get("access_token")
    if not access_token:
        flash("카카오 로그인 실패: access_token이 없습니다.", "error")
        return redirect(url_for("auth.login"))

    # 2) 사용자 정보 요청
    try:
        user_res = requests.get(
            "https://kapi.kakao.com/v2/user/me",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        user_res.raise_for_status()
    except requests.RequestException:
        current_app.logger.exception("카카오 사용자 정보 요청 실패")
        flash("카카오 로그인 중 오류가 발생했습니다. (user)", "error")
        return redirect(url_for("auth.login"))

    user_info = user_res.json()
    kakao_id = user_info.get("id")
    kakao_account = user_info.get("kakao_account", {}) or {}
    email = kakao_account.get("email")
    profile = kakao_account.get("profile") or {}
    nickname = profile.get("nickname", "카카오사용자")

    users_col = get_users_col()

    # 3) DB에서 유저 찾기 (kakao_id 우선, 없으면 email로도 한 번 더)
    user = None
    if kakao_id:
        user = users_col.find_one({"kakao_id": kakao_id})
    if not user and email:
        user = users_col.find_one({"email": email})

    # 4) 없으면 새로 생성
    if not user:
        pw_hash = generate_password_hash("KAKAO_LOGIN_ONLY")  # 실제로 쓰지 않을 더미 패스워드
        user_doc = {
            "email": email,
            "password_hash": pw_hash,
            "name": nickname,
            "kakao_id": kakao_id,
        }
        result = users_col.insert_one(user_doc)
        user_id = result.inserted_id
        user_email = email
        user_name = nickname
    else:
        user_id = user["_id"]
        user_email = user.get("email")
        user_name = user.get("name", nickname)

        # 기존 유저에 kakao_id가 없으면 업데이트
        if kakao_id and not user.get("kakao_id"):
            users_col.update_one(
                {"_id": user_id},
                {"$set": {"kakao_id": kakao_id}}
            )

    # 5) 세션에 로그인 정보 넣기 (일반 로그인과 동일한 형식)
    session["user_id"] = str(user_id)
    session["user_email"] = user_email
    session["user_name"] = user_name

    flash(f"{user_name}님, 카카오 계정으로 로그인되었습니다.", "success")
    return redirect(url_for("main.ev_page"))

# ---------------- 네이버 로그인 ----------------

@auth_bp.route("/login/naver")
def naver_login():
    naver_client_id = current_app.config.get("NAVER_CLIENT_ID")
    redirect_uri = current_app.config.get("NAVER_REDIRECT_URI")

    if not naver_client_id or not redirect_uri:
        flash("네이버 로그인 설정이 없습니다. 관리자에게 문의해 주세요.", "error")
        return redirect(url_for("auth.login"))

    state = secrets.token_urlsafe(16)
    session["naver_oauth_state"] = state

    naver_auth_url = (
        "https://nid.naver.com/oauth2.0/authorize"
        f"?response_type=code"
        f"&client_id={naver_client_id}"
        f"&redirect_uri={redirect_uri}"
        f"&state={state}"
    )
    return redirect(naver_auth_url)


@auth_bp.route("/login/naver/callback")
def naver_callback():
    error = request.args.get("error")
    if error:
        flash("네이버 로그인 중 사용자가 취소했거나 오류가 발생했습니다.", "error")
        return redirect(url_for("auth.login"))

    code = request.args.get("code")
    state = request.args.get("state")
    state_in_session = session.pop("naver_oauth_state", None)

    if not code or not state or state != state_in_session:
        flash("네이버 로그인 요청 상태가 올바르지 않습니다.", "error")
        return redirect(url_for("auth.login"))

    # 1) 액세스 토큰 요청
    token_res = requests.post(
        "https://nid.naver.com/oauth2.0/token",
        data={
            "grant_type": "authorization_code",
            "client_id": current_app.config["NAVER_CLIENT_ID"],
            "client_secret": current_app.config["NAVER_CLIENT_SECRET"],
            "redirect_uri": current_app.config["NAVER_REDIRECT_URI"],
            "code": code,
            "state": state,
        },
    )
    try:
        token_res.raise_for_status()
    except requests.RequestException:
        current_app.logger.exception("네이버 토큰 요청 실패")
        flash("네이버 로그인 중 오류가 발생했습니다. (token)", "error")
        return redirect(url_for("auth.login"))

    token_json = token_res.json()
    access_token = token_json.get("access_token")
    if not access_token:
        flash("네이버 로그인 실패: access_token이 없습니다.", "error")
        return redirect(url_for("auth.login"))

    # 2) 사용자 정보 요청
    try:
        user_res = requests.get(
            "https://openapi.naver.com/v1/nid/me",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        user_res.raise_for_status()
    except requests.RequestException:
        current_app.logger.exception("네이버 사용자 정보 요청 실패")
        flash("네이버 로그인 중 오류가 발생했습니다. (user)", "error")
        return redirect(url_for("auth.login"))

    info = user_res.json()
    resp = info.get("response", {}) or {}
    naver_id = resp.get("id")
    email = resp.get("email")
    name = resp.get("name") or "Naver 사용자"

    users_col = get_users_col()

    # DB에서 유저 찾기
    user = None
    if naver_id:
        user = users_col.find_one({"naver_id": naver_id})
    if not user and email:
        user = users_col.find_one({"email": email})

    # 없으면 새로 생성
    if not user:
        pw_hash = generate_password_hash("NAVER_LOGIN_ONLY")
        user_doc = {
            "email": email,
            "password_hash": pw_hash,
            "name": name,
            "naver_id": naver_id,
        }
        result = users_col.insert_one(user_doc)
        user_id = result.inserted_id
        user_email = email
        user_name = name
    else:
        user_id = user["_id"]
        user_email = user.get("email")
        user_name = user.get("name", name)
        if naver_id and not user.get("naver_id"):
            users_col.update_one(
                {"_id": user_id},
                {"$set": {"naver_id": naver_id}}
            )

    # 세션 로그인 처리 (카카오랑 동일한 형식)
    session["user_id"] = str(user_id)
    session["user_email"] = user_email
    session["user_name"] = user_name

    flash(f"{user_name}님, 네이버 계정으로 로그인되었습니다.", "success")
    return redirect(url_for("main.ev_page"))