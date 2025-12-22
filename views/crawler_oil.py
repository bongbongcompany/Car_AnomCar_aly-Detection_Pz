# App/views/crawler_oil.py

import json
from pathlib import Path
from datetime import datetime, timedelta

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# ───────────────── 기본 설정 ─────────────────
URL_OIL = "https://www.uga.go.kr/pbc/taoi/view"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36"
)

# 24시간 캐시
CACHE_FILE = Path(__file__).with_name("oil_cache.json")
CACHE_TTL = timedelta(hours=24)   # 24시간 동안 재사용


# ──────────────── 실제 크롤링 (Selenium) ────────────────
def crawl_oil_prices():
    """
    유가보조금 포털에서 '평균유가' 카드(경유, LPG, 수소, CNG 등)를 크롤링해서
    {'date': '...', 'items': [...]} 형태로 반환.
    DB 사용 안 함.
    """

    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    # 헤더(User-Agent) 우회
    chrome_options.add_argument(f"--user-agent={USER_AGENT}")
    chrome_options.add_argument("--lang=ko_KR.UTF-8")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # 1) 페이지 접속
        driver.get(URL_OIL)

        wait = WebDriverWait(driver, 10)

        # 날짜/지역 표시 요소 로딩 대기
        wait.until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "p.oilps-info-date")
            )
        )

        # 2) 날짜/지역 정보
        date_text = driver.find_element(
            By.CSS_SELECTOR, "p.oilps-info-date"
        ).text.strip()

        # 3) 유종 별 카드들
        boxes = driver.find_elements(
            By.CSS_SELECTOR,
            "div.oilps-map-info-box-group div.oilps-map-info-box"
        )

        items = []
        for box in boxes:
            fuel = box.find_element(
                By.CSS_SELECTOR, "p.oilps-info-con-title"
            ).text.strip()          # 예: "경유 (ℓ)"

            price = box.find_element(
                By.CSS_SELECTOR, "p.oilps-info-con-value"
            ).text.strip()          # 예: "1,662.89원"

            items.append({
                "fuel": fuel,
                "price": price,
            })

        return {
            "date": date_text,
            "items": items,
        }

    finally:
        driver.quit()


# ──────────────── 캐시 유틸 ────────────────
def _load_cache():
    """24시간 이내 캐시가 있으면 그걸 반환."""
    if not CACHE_FILE.exists():
        return None
    try:
        raw = CACHE_FILE.read_text(encoding="utf-8")
        data = json.loads(raw)

        fetched_at = data.get("fetched_at")
        if not fetched_at:
            return None

        ts = datetime.fromisoformat(fetched_at)
        if datetime.now() - ts > CACHE_TTL:
            # 캐시 만료
            return None

        payload = data.get("payload")
        # items 가 비어 있으면 캐시 사용 안 함
        if not payload or not payload.get("items"):
            return None

        return payload
    except Exception:
        return None


def _save_cache(payload: dict):
    """현재 결과를 캐시 파일에 저장."""
    data = {
        "fetched_at": datetime.now().isoformat(),
        "payload": payload,
    }
    CACHE_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ──────────────── 외부에서 쓰는 함수 ────────────────
def get_oil_prices():
    """
    1) 캐시에 유효한 값 있으면 그대로 반환
    2) 아니면 Selenium으로 크롤링 후, 24시간 캐시 파일에 저장
    """
    cached = _load_cache()
    if cached is not None:
        return cached

    payload = crawl_oil_prices()

    if payload.get("items"):
        _save_cache(payload)

    return payload


# 단독 실행 테스트용
if __name__ == "__main__":
    from pprint import pprint
    pprint(get_oil_prices())
