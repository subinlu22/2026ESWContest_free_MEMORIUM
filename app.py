import os
import json
import shutil
import time
import unicodedata
import requests
import cv2
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, Response
from werkzeug.utils import secure_filename

load_dotenv()  # .env 파일에서 GEMINI_API_KEY 등 환경변수 불러오기 (git에는 안 올라감)

# =============================================
# 메모리움 웹 설정 페이지
# =============================================

app = Flask(__name__)
app.secret_key = "memorium_secret"

# ── 경로 설정 ──────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))  # 이 스크립트 파일이 있는 폴더를 자동으로 잡음 (윈도우/리눅스 둘 다 동작)
PHOTOS_DIR    = os.path.join(BASE_DIR, "photos")
VIDEOS_DIR    = os.path.join(BASE_DIR, "videos")
MUSIC_DIR     = os.path.join(BASE_DIR, "music")
THUMB_DIR     = os.path.join(BASE_DIR, "video_thumbs")  # [신규] 영상 첫 프레임 썸네일 캐시
MEMORY_FILE   = os.path.join(BASE_DIR, "memory.json")
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")
TAGS_FILE     = os.path.join(BASE_DIR, "tags.json")
VIEWING_LOG_FILE = os.path.join(BASE_DIR, "viewing_log.json")
SPEECH_LOG_FILE  = os.path.join(BASE_DIR, "speech_log.json")
AI_SUMMARY_FILE  = os.path.join(BASE_DIR, "ai_summary.json")  # [신규] 오늘의 활동 통계(AI 요약) 캐시

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

ALLOWED_IMAGES = {"jpg", "jpeg", "png"}
ALLOWED_VIDEOS = {"mp4", "avi", "mov"}
ALLOWED_MUSIC  = {"mp3", "wav"}

DEFAULT_SETTINGS = {
    "medicine_times": [{"hour": 14, "minute": 0}],
    "chime_enabled": True,
    "sundowner_enabled": True,
    "sundowner_hour": 19,
    "sundowner_minute": 0,
    "sundowner_end_hour": 21,
    "sundowner_end_minute": 0,
    "sundowner_auto_music": True,    # 썬다운 시간대에 자동으로 차분한 음악 재생할지
    "face_auto_music": True,         # 평소 시간대에 얼굴 인식 시 자동으로 음악 재생할지
    "slide_interval": 10
}


# =============================================
# 유틸 함수
# =============================================
_weather_cache = {"data": None, "ts": 0}
WEATHER_LAT = 37.5665   # 서울 기준 (액자가 다른 지역이면 이 값만 바꾸면 됨)
WEATHER_LON = 126.9780

def build_daily_summary_prompt(viewing_stats, speech_summary, top_tags, profile_name=None):
    """오늘 데이터를 모아 Gemini에게 보낼 프롬프트 구성"""
    today_minutes = viewing_stats["today_minutes"]
    today_count = viewing_stats["today_count"]
    speech_count = speech_summary.get("count", 0)
    top_words = speech_summary.get("top_words", [])
    top_tag = top_tags[0]["tag"] if top_tags else None
    top_pct = top_tags[0]["pct"] if top_tags else None

    lines = [
        "너는 치매 초기 어르신을 돌보는 스마트 액자 '메모리움'의 보호자용 하루 요약을 써주는 도우미야.",
        "아래 오늘 하루 데이터를 바탕으로, 보호자가 읽었을 때 안심되고 따뜻한 느낌을 받을 수 있는 한국어 요약을 딱 2문장으로 써줘.",
        "평가하거나 걱정하는 말투 대신, 오늘 있었던 일을 다정하게 전달하는 톤으로 써줘.",
        "'~하셨어요', '~보였어요' 같은 존댓말을 쓰고, 숫자를 그냥 나열하기보다는 자연스러운 문장으로 녹여줘.",
    ]
    if profile_name:
        lines.append(f"사람을 지칭할 때는 '어르신'이라는 말 대신 반드시 '{profile_name}님'이라고 불러줘.")
    else:
        lines.append("사람을 지칭할 때 '어르신'이라는 말은 쓰지 말고, 주어를 자연스럽게 생략하거나 풀어서 써줘.")
    lines += [
        "",
        f"- 오늘 액자를 본 횟수: {today_count}번",
        f"- 오늘 액자를 본 시간: {today_minutes}분",
    ]
    if top_tag:
        lines.append(f"- 오늘 가장 많이 반응한 사진 주제: {top_tag} ({top_pct}%)")
    if speech_count:
        lines.append(f"- 오늘 말씀하신 횟수: {speech_count}회")
    if top_words:
        words_str = ", ".join([w for w, c in top_words[:3]])
        lines.append(f"- 오늘 자주 하신 말: {words_str}")
    if today_count == 0 and speech_count == 0:
        lines.append("- 오늘은 아직 기록된 활동이 없음 (이 경우 '아직 하루가 시작 전이에요' 느낌으로 짧게)")

    lines.append("")
    lines.append("2문장 요약만 출력하고, 따옴표나 다른 설명은 붙이지 마.")
    return "\n".join(lines)


def get_daily_ai_summary(viewing_stats, speech_summary, top_tags, profile_name=None):
    """
    오늘 하루 데이터를 Gemini API로 요약해서 반환
    - ai_summary.json에 캐시해두고, 같은 날짜면 30분에 한 번만 새로 생성
      (페이지 열 때마다 API 호출하면 비용/속도 낭비라서)
    - API 키가 없거나 호출 실패하면 None(또는 예전 캐시) 반환
      → 템플릿에서 "준비 중" 안내 문구로 자연스럽게 대체됨
    """
    import datetime as dt
    today_str = dt.datetime.now().strftime("%Y-%m-%d")

    cache = {}
    if os.path.exists(AI_SUMMARY_FILE):
        try:
            with open(AI_SUMMARY_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    now_ts = time.time()
    if (cache.get("date") == today_str
            and cache.get("text")
            and now_ts - cache.get("generated_at", 0) < 1800):
        return cache["text"]

    if not GEMINI_API_KEY:
        return None

    prompt = build_daily_summary_prompt(viewing_stats, speech_summary, top_tags, profile_name)

    try:
        resp = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
            params={"key": GEMINI_API_KEY},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=10,
        )
        data = resp.json()
    except Exception as e:
        print(f"⚠️ 오늘의 활동 통계(Gemini) 요청 자체 실패(네트워크 등): {e}")
        return cache.get("text")

    if "candidates" not in data:
        # 여기가 이번에 진짜 원인이 찍히는 부분 — status_code랑 data 그대로 출력
        print(f"⚠️ 오늘의 활동 통계(Gemini) API 에러 응답 (status={resp.status_code}): {data}")
        return cache.get("text")

    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"⚠️ 오늘의 활동 통계(Gemini) 응답 파싱 실패: {e} / data={data}")
        return cache.get("text")

    with open(AI_SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump({"date": today_str, "generated_at": now_ts, "text": text}, f, ensure_ascii=False, indent=2)

    return text


def get_current_weather():
    """
    Open-Meteo 무료 API로 현재 날씨 가져오기 (API 키 불필요)
    - 30분 캐시해서 대시보드 새로고침마다 매번 호출하지 않게 함
    - 네트워크 문제 등으로 실패하면 None 반환 (템플릿에서 날씨 부분만 조용히 숨김)
    """
    now = time.time()
    if _weather_cache["data"] and now - _weather_cache["ts"] < 1800:
        return _weather_cache["data"]

    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": WEATHER_LAT,
                "longitude": WEATHER_LON,
                "current_weather": "true",
                "timezone": "Asia/Seoul",
            },
            timeout=3,
        )
        cw = resp.json()["current_weather"]
        code = cw["weathercode"]
        temp = round(cw["temperature"])

        if code == 0:
            icon, desc = "☀", "맑음"
        elif code in (1, 2, 3):
            icon, desc = "☁", "흐림"
        elif code in (45, 48):
            icon, desc = "▦", "안개"
        elif code in (51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82):
            icon, desc = "☔", "비"
        elif code in (71, 73, 75, 77, 85, 86):
            icon, desc = "❄", "눈"
        elif code in (95, 96, 99):
            icon, desc = "☈", "뇌우"
        else:
            icon, desc = "☁", "흐림"

        data = {"temp": temp, "desc": desc, "icon": icon}
        _weather_cache["data"] = data
        _weather_cache["ts"] = now
        return data
    except Exception:
        return None


def allowed_file(filename, allowed):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_SETTINGS.copy()

def save_settings(s):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_memory(m):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)

def load_tags():
    if os.path.exists(TAGS_FILE):
        with open(TAGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_tags(t):
    with open(TAGS_FILE, "w", encoding="utf-8") as f:
        json.dump(t, f, ensure_ascii=False, indent=2)

def load_viewing_log():
    """viewing_log.json에서 응시 기록 목록 불러오기"""
    if os.path.exists(VIEWING_LOG_FILE):
        with open(VIEWING_LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def load_speech_log():
    """speech_log.json에서 말씀 텍스트 기록 불러오기 — 최신순(저장순) 20개"""
    if os.path.exists(SPEECH_LOG_FILE):
        with open(SPEECH_LOG_FILE, "r", encoding="utf-8") as f:
            log = json.load(f)
        # 저장된 순서(인덱스)가 곧 시간순 — 인덱스 역순으로 정렬해서 최신이 위로
        return list(reversed(log))[:20]
    return []

def get_today_speech_summary():
    """
    오늘 기록된 말씀(speech_log)을 분석해서 요약 정보 반환
    - 오늘 몇 번 말씀하셨는지
    - 자주 나온 단어 Top 5 (조사/의미없는 한 글자 단어는 제외)
    기존 load_speech_log()는 최신 20개로 제한돼 있어서(날짜 안 가림),
    "오늘"만 정확히 걸러내기 위해 파일을 따로 읽음 — 기존 함수는 그대로 둠
    """
    import datetime as dt
    from collections import Counter

    if not os.path.exists(SPEECH_LOG_FILE):
        return {"count": 0, "top_words": []}

    with open(SPEECH_LOG_FILE, "r", encoding="utf-8") as f:
        log = json.load(f)

    today_str = dt.datetime.now().strftime("%Y-%m-%d")
    today_records = [r for r in log if r.get("date") == today_str]

    # 조사/의미 없는 한 글자 단어는 분석에서 제외
    STOPWORDS = {"이", "가", "은", "는", "을", "를", "에", "의", "도", "요", "그", "저", "나", "너"}
    words = []
    for r in today_records:
        for w in r.get("text", "").split():
            w = w.strip("().,!?~ㅋ ")
            if len(w) >= 2 and w not in STOPWORDS:
                words.append(w)

    top_words = Counter(words).most_common(5)

    return {"count": len(today_records), "top_words": top_words}


def get_today_sessions():
    """
    오늘 응시 기록을 24시간 원형 시계 위의 점 좌표로 변환해서 반환
    - 12시(자정) 방향이 원 맨 위(0도), 시계방향으로 24시간 회전
    - SVG viewBox 200x200, 중심(100,100), 반지름 80 기준으로 계산
    - 프론트엔드(dashboard.html)에서 각 점을 <circle>로 그리는 용도
    """
    import math

    log = load_viewing_log()
    today_str = __import__("datetime").datetime.now().strftime("%Y-%m-%d")
    today_records = [r for r in log if r["date"] == today_str]

    RADIUS = 80
    CENTER = 100

    sessions = []
    for r in today_records:
        try:
            h, m, s = map(int, r["start"].split(":"))
        except Exception:
            continue
        hour_float = h + m / 60 + s / 3600
        angle_deg = (hour_float / 24) * 360 - 90   # -90도: 0시를 원 맨 위(12시 방향)로 맞춤
        angle_rad = math.radians(angle_deg)
        x = round(CENTER + RADIUS * math.cos(angle_rad), 1)
        y = round(CENTER + RADIUS * math.sin(angle_rad), 1)
        sessions.append({
            "x": x, "y": y,
            "start": r["start"][:5],
            "duration": r["duration_minutes"]
        })
    return sessions


def get_activity_donut_gradient(sessions):
    """
    오늘 응시 세션을 24시간 링(conic-gradient) 색상 문자열로 변환
    - 실제로 본 시간대만 초록색, 나머지는 연한 회색
    - get_today_sessions()와 동일하게 0시=원 맨 위, 시계방향 기준
    """
    if not sessions:
        return "#e9eee5 0% 100%"

    intervals = []
    for s in sessions:
        try:
            h, m = map(int, s["start"].split(":"))
        except Exception:
            continue
        start_pct = (h * 60 + m) / 1440 * 100
        dur_pct = max(s.get("duration", 0), 0) / 1440 * 100
        end_pct = min(start_pct + dur_pct, 100)
        if end_pct > start_pct:
            intervals.append((start_pct, end_pct))

    if not intervals:
        return "#e9eee5 0% 100%"

    intervals.sort()
    # 겹치거나 거의 붙어있는 구간은 하나로 병합 (0.3%p ≈ 4분 정도 이내면 붙임)
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1] + 0.3:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    stops = []
    cursor = 0.0
    for start, end in merged:
        if start > cursor:
            stops.append(f"#e9eee5 {cursor:.2f}% {start:.2f}%")
        stops.append(f"#2c6231 {start:.2f}% {end:.2f}%")
        cursor = end
    if cursor < 100:
        stops.append(f"#e9eee5 {cursor:.2f}% 100%")

    return ", ".join(stops)


def get_today_activity_facts(sessions):
    """
    오늘 응시 세션 목록(get_today_sessions 결과)을 바탕으로
    '가장 많이 본 시간대'와 '마지막 시청 시각'을 계산
    - 새 대시보드 디자인의 "오늘의 활동" 카드에서 사용
    - sessions가 비어있으면 None 반환 (템플릿에서 '기록 없음' 처리)
    """
    from collections import Counter

    if not sessions:
        return {"most_hour_range": None, "last_time": None}

    def fmt(h, m=None):
        period = "오전" if h < 12 else "오후"
        h12 = h % 12
        if h12 == 0:
            h12 = 12
        return f"{period} {h12}시" + (f" {m}분" if m is not None and m > 0 else "")

    hours = [int(s["start"].split(":")[0]) for s in sessions]
    most_hour = Counter(hours).most_common(1)[0][0]
    most_hour_range = f"{fmt(most_hour)} ~ {fmt((most_hour + 1) % 24)}"

    last = max(sessions, key=lambda s: s["start"])
    h, m = map(int, last["start"].split(":")[:2])
    last_time = fmt(h, m)

    return {"most_hour_range": most_hour_range, "last_time": last_time}


def get_viewing_stats():
    """
    응시 기록을 바탕으로 요약 통계 계산
    - 오늘 응시 횟수, 오늘 총 응시 시간
    - 전체 응시 횟수, 전체 총 응시 시간
    - 최근 기록 10개 (최신순)
    """
    import datetime as dt
    log = load_viewing_log()
    today_str = dt.datetime.now().strftime("%Y-%m-%d")

    today_records = [r for r in log if r["date"] == today_str]
    today_count    = len(today_records)
    today_minutes  = round(sum(r["duration_minutes"] for r in today_records), 1)

    total_count   = len(log)
    total_minutes = round(sum(r["duration_minutes"] for r in log), 1)

    recent = sorted(log, key=lambda r: (r["date"], r["start"]), reverse=True)[:10]

    return {
        "today_count": today_count,
        "today_minutes": today_minutes,
        "total_count": total_count,
        "total_minutes": total_minutes,
        "recent": recent
    }

def get_categories():
    """photos 하위 폴더 목록 반환 (카테고리)"""
    if not os.path.exists(PHOTOS_DIR):
        return []
    return [d for d in os.listdir(PHOTOS_DIR)
            if os.path.isdir(os.path.join(PHOTOS_DIR, d))]

def get_photos_in_category(category):
    """카테고리 폴더 내 사진 목록 반환"""
    folder = os.path.join(PHOTOS_DIR, category)
    if not os.path.exists(folder):
        return []
    return [f for f in os.listdir(folder)
            if f.rsplit(".", 1)[-1].lower() in ALLOWED_IMAGES]

def get_video_categories():
    """videos 하위 폴더 목록 반환"""
    if not os.path.exists(VIDEOS_DIR):
        return []
    return [d for d in os.listdir(VIDEOS_DIR)
            if os.path.isdir(os.path.join(VIDEOS_DIR, d))]

def get_videos_in_category(category):
    """영상 카테고리 폴더 내 파일 목록 반환"""
    folder = os.path.join(VIDEOS_DIR, category)
    if not os.path.exists(folder):
        return []
    return [f for f in os.listdir(folder)
            if f.rsplit(".", 1)[-1].lower() in ALLOWED_VIDEOS]

def get_file_list(folder, extensions):
    if not os.path.exists(folder):
        return []
    return [f for f in os.listdir(folder)
            if f.rsplit(".", 1)[-1].lower() in extensions]


# =============================================
# 대시보드 (홈)
# =============================================
@app.route("/")
def dashboard():
    import datetime as dt

    viewing_stats = get_viewing_stats()
    speech_log    = load_speech_log()[:6]
    today_sessions = get_today_sessions()
    speech_summary = get_today_speech_summary()  # [신규] 오늘 하신 말씀 횟수 카드용 — 기존 함수 그대로 재사용
    activity_facts = get_today_activity_facts(today_sessions)  # [신규] 가장 많이 본 시간/마지막 시청 카드용
    activity_gradient = get_activity_donut_gradient(today_sessions)  # [신규] 오늘의 활동 도넛 실데이터 색상

    mem = load_memory()
    total = sum(mem.values()) if mem else 0
    sorted_mem = sorted(mem.items(), key=lambda x: -x[1])

    # [신규] 태그당 대표 사진/영상 찾기 — "많이 본 사진 주제", "기억에 반응한 주제" 썸네일용
    tags_data = load_tags()

    def find_representative_media(tag_name):
        """
        우선순위:
          1. tags.json에 이 태그가 명시적으로 달린 사진
          2. tags.json에 이 태그가 명시적으로 달린 영상
          3. 태그명 == 사진 폴더명인 경우 - 그 폴더의 첫 사진
          4. 태그명 == 영상 폴더명인 경우 - 그 폴더의 첫 영상
        한글은 자모 정규화 방식(NFC/NFD)이 폴더/텍스트마다 다를 수 있어서
        비교 전에 항상 NFC로 맞춰서 비교함 (안 그러면 눈에는 같아 보여도 안 걸림)
        """
        target = unicodedata.normalize("NFC", tag_name)

        photo_match, video_match = None, None
        for key, tag_list in tags_data.items():
            normalized_tags = [unicodedata.normalize("NFC", t) for t in tag_list]
            if target not in normalized_tags:
                continue
            if key.startswith("v:"):
                real_key = key[2:]
                if not video_match and os.path.exists(os.path.join(VIDEOS_DIR, real_key)):
                    video_match = real_key
            else:
                if not photo_match and os.path.exists(os.path.join(PHOTOS_DIR, key)):
                    photo_match = key
        if photo_match:
            return {"type": "photo", "key": photo_match}
        if video_match:
            return {"type": "video", "key": video_match}

        for cat in get_categories():
            if unicodedata.normalize("NFC", cat) == target:
                photos = get_photos_in_category(cat)
                if photos:
                    return {"type": "photo", "key": f"{cat}/{photos[0]}"}

        for cat in get_video_categories():
            if unicodedata.normalize("NFC", cat) == target:
                videos = get_videos_in_category(cat)
                if videos:
                    return {"type": "video", "key": f"{cat}/{videos[0]}"}

        # [디버그] 왜 못 찾았는지 터미널에 그대로 출력 (repr로 찍어서 안 보이는 공백/문자까지 다 드러남)
        print(f"🔍 미디어 매칭 실패 - 태그: {tag_name!r} (정규화: {target!r})")
        print(f"    사진 카테고리: {[c for c in get_categories()]!r}")
        print(f"    영상 카테고리: {[c for c in get_video_categories()]!r}")
        return None

    top_tags = []
    for tag, count in sorted_mem[:10]:
        media = find_representative_media(tag)
        photo_url = None
        video_thumb_url = None
        if media and media["type"] == "photo":
            photo_url = url_for("serve_photo", filename=media["key"])
        elif media and media["type"] == "video":
            # 미리 추출 시도 — 성공한 경우에만 URL을 넣어서, 실패해도 화면엔 회색 박스로 자연스럽게 폴백
            if get_or_create_video_thumb(media["key"]):
                video_thumb_url = url_for("video_thumbnail", filename=media["key"])
        top_tags.append({
            "tag": tag,
            "count": count,
            "pct": round(count / total * 100, 1) if total > 0 else 0,
            "photo_url": photo_url,
            "video_thumb_url": video_thumb_url,
        })

    now = dt.datetime.now()
    weekday_kr = ["월", "화", "수", "목", "금", "토", "일"][now.weekday()]

    # [신규] 액자 프로필(이름/사진) — 가입 계정 없이 settings.json에 저장해두고 불러오는 방식
    s = load_settings()
    profile_name = s.get("profile_name", "")
    profile_photo = s.get("profile_photo", "")
    profile_photo_url = url_for("static", filename=f"profile/{profile_photo}") if profile_photo else None

    # [신규] 오늘의 날씨
    weather = get_current_weather()

    # [신규] 오늘의 활동 통계 (Gemini API 요약)
    ai_summary = get_daily_ai_summary(viewing_stats, speech_summary, top_tags, profile_name)

    # [신규] "최근 활동" 썸네일 — viewing_log에 media_key가 있는 최신 기록부터 실제 사진/영상 연결
    # (media_key는 slideshow_tts.py를 업데이트해야 새로 쌓이는 기록부터 생김 —
    #  그 전 기록에는 없을 수 있어서 없으면 조용히 placeholder로 처리)
    def resolve_media_url(media_key):
        if not media_key:
            return None, None
        if media_key.startswith("v:"):
            video_key = media_key[2:]
            if get_or_create_video_thumb(video_key):
                return None, url_for("video_thumbnail", filename=video_key)
            return None, None
        # 사진도 실제로 그 경로에 파일이 있는지 확인 후에만 URL 반환
        # (media_key가 어떤 이유로든 폴더 없이 파일명만 기록됐거나 잘못됐을 경우 대비)
        if os.path.exists(os.path.join(PHOTOS_DIR, media_key)):
            return url_for("serve_photo", filename=media_key), None
        print(f"⚠️ 최근 활동 사진 매칭 실패 - media_key: {media_key!r} (경로에 파일 없음)")
        return None, None

    recent_with_media = []
    for r in viewing_stats["recent"][:3]:
        photo_url, video_thumb_url = resolve_media_url(r.get("media_key"))
        recent_with_media.append({**r, "photo_url": photo_url, "video_thumb_url": video_thumb_url})

    return render_template("dashboard.html",
                           viewing=viewing_stats,
                           speech=speech_log,
                           top_tags=top_tags,
                           today_sessions=today_sessions,
                           speech_summary=speech_summary,
                           activity_facts=activity_facts,
                           activity_gradient=activity_gradient,
                           profile_name=profile_name,
                           profile_photo_url=profile_photo_url,
                           weather=weather,
                           ai_summary=ai_summary,
                           recent_with_media=recent_with_media,
                           today_date=now.strftime("%Y년 %m월 %d일"),
                           today_weekday=weekday_kr)


# =============================================
# 미디어 관리 페이지
# =============================================
@app.route("/media", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        action = request.form.get("action")

        # ── 카테고리 추가 ──
        if action == "add_category":
            name = request.form.get("category_name", "").strip()
            if name:
                path = os.path.join(PHOTOS_DIR, name)
                os.makedirs(path, exist_ok=True)
                flash(f"✅ '{name}' 카테고리 추가 완료!", "success")
            return redirect(url_for("index"))

        # ── 카테고리 삭제 ──
        elif action == "delete_category":
            name = request.form.get("category_name", "").strip()
            path = os.path.join(PHOTOS_DIR, name)
            if os.path.exists(path):
                shutil.rmtree(path)
                # 해당 카테고리 사진 태그도 삭제
                tags = load_tags()
                to_delete = [k for k in tags if k.startswith(f"{name}/")]
                for k in to_delete:
                    del tags[k]
                save_tags(tags)
                flash(f"✅ '{name}' 카테고리 삭제 완료!", "success")
            return redirect(url_for("index"))

        # ── 사진 업로드 ──
        elif action == "upload_photos":
            category   = request.form.get("category", "").strip()
            extra_tags = [t.strip() for t in request.form.getlist("extra_tags") if t.strip()]
            new_tag    = request.form.get("new_tag", "").strip()
            if new_tag:
                extra_tags.append(new_tag)
            files = request.files.getlist("files")

            if not category:
                flash("카테고리를 선택해주세요.", "error")
                return redirect(url_for("index"))

            folder = os.path.join(PHOTOS_DIR, category)
            os.makedirs(folder, exist_ok=True)
            tags = load_tags()
            success, fail = 0, 0

            for file in files:
                if file and allowed_file(file.filename, ALLOWED_IMAGES):
                    filename = secure_filename(file.filename)
                    file.save(os.path.join(folder, filename))
                    # 태그 키: "카테고리/파일명"
                    key = f"{category}/{filename}"
                    tags[key] = [category] + extra_tags  # 폴더명이 1차 태그
                    success += 1
                else:
                    fail += 1

            save_tags(tags)
            if success:
                flash(f"✅ {success}개 업로드 완료!", "success")
            if fail:
                flash(f"❌ {fail}개 실패", "error")
            return redirect(url_for("index"))

        # ── 영상 카테고리 추가 ──
        elif action == "add_video_category":
            name = request.form.get("category_name", "").strip()
            if name:
                os.makedirs(os.path.join(VIDEOS_DIR, name), exist_ok=True)
                flash(f"✅ '{name}' 영상 카테고리 추가 완료!", "success")
            return redirect(url_for("index"))

        # ── 영상 카테고리 삭제 ──
        elif action == "delete_video_category":
            name = request.form.get("category_name", "").strip()
            path = os.path.join(VIDEOS_DIR, name)
            if os.path.exists(path):
                shutil.rmtree(path)
                tags = load_tags()
                to_delete = [k for k in tags if k.startswith(f"v:{name}/")]
                for k in to_delete:
                    del tags[k]
                save_tags(tags)
                flash(f"✅ '{name}' 영상 카테고리 삭제 완료!", "success")
            return redirect(url_for("index"))

        # ── 영상 업로드 ──
        elif action == "upload_videos":
            category   = request.form.get("category", "").strip()
            extra_tags = [t.strip() for t in request.form.getlist("extra_tags") if t.strip()]
            new_tag    = request.form.get("new_tag", "").strip()
            if new_tag:
                extra_tags.append(new_tag)
            files = request.files.getlist("files")

            if not category:
                flash("카테고리를 선택해주세요.", "error")
                return redirect(url_for("index"))

            folder = os.path.join(VIDEOS_DIR, category)
            os.makedirs(folder, exist_ok=True)
            tags = load_tags()
            success, fail = 0, 0

            for file in files:
                if file and allowed_file(file.filename, ALLOWED_VIDEOS):
                    filename = secure_filename(file.filename)
                    file.save(os.path.join(folder, filename))
                    key = f"v:{category}/{filename}"  # v: 접두사로 영상 구분
                    tags[key] = [category] + extra_tags
                    success += 1
                else:
                    fail += 1

            save_tags(tags)
            if success:
                flash(f"✅ {success}개 업로드 완료!", "success")
            return redirect(url_for("index"))

        # ── 음악 업로드 ──
        elif action == "upload_music":
            files = request.files.getlist("files")
            os.makedirs(MUSIC_DIR, exist_ok=True)
            success = 0
            for file in files:
                if file and allowed_file(file.filename, ALLOWED_MUSIC):
                    file.save(os.path.join(MUSIC_DIR, secure_filename(file.filename)))
                    success += 1
            if success:
                flash(f"✅ {success}개 업로드 완료!", "success")
            return redirect(url_for("index"))

        # ── 태그 편집 ──
        elif action == "update_tags":
            key       = request.form.get("key")
            tag_names = [t.strip() for t in request.form.getlist("tags") if t.strip()]
            new_tag   = request.form.get("new_tag", "").strip()
            if new_tag:
                tag_names.append(new_tag)
            tags = load_tags()
            if tag_names:
                tags[key] = tag_names
            elif key in tags:
                del tags[key]
            save_tags(tags)
            flash("✅ 태그 업데이트 완료!", "success")
            return redirect(url_for("index"))

        # ── 파일 삭제 ──
        elif action == "delete_file":
            key    = request.form.get("key")
            folder = request.form.get("folder")

            if folder == "photos":
                path = os.path.join(PHOTOS_DIR, key)
            elif folder == "videos":
                # key = "v:카테고리/파일명"
                real_key = key.replace("v:", "", 1)
                path = os.path.join(VIDEOS_DIR, real_key)
            elif folder == "music":
                path = os.path.join(MUSIC_DIR, key)
            else:
                return redirect(url_for("index"))

            if os.path.exists(path):
                os.remove(path)
                tags = load_tags()
                if key in tags:
                    del tags[key]
                save_tags(tags)
                flash("✅ 삭제 완료!", "success")
            return redirect(url_for("index"))

    # 데이터 조회
    categories       = get_categories()
    category_photos  = {cat: get_photos_in_category(cat) for cat in categories}
    video_categories = get_video_categories()
    category_videos  = {cat: get_videos_in_category(cat) for cat in video_categories}
    music            = get_file_list(MUSIC_DIR, ALLOWED_MUSIC)
    tags             = load_tags()
    all_tags         = sorted(set(t for tlist in tags.values() for t in tlist))

    return render_template("index.html",
                           categories=categories,
                           category_photos=category_photos,
                           video_categories=video_categories,
                           category_videos=category_videos,
                           music=music,
                           tags=tags,
                           all_tags=all_tags)


# =============================================
# 설정 페이지
# =============================================
# =============================================
# 프로필(액자 이름/사진) 저장 — 가입/로그인 없이 settings.json에 저장
# =============================================
@app.route("/profile", methods=["POST"])
def update_profile():
    s = load_settings()

    name = request.form.get("profile_name", "").strip()
    if name:
        s["profile_name"] = name

    photo = request.files.get("profile_photo")
    if photo and photo.filename:
        ext = photo.filename.rsplit(".", 1)[-1].lower()
        if ext in ALLOWED_IMAGES:
            profile_dir = os.path.join(BASE_DIR, "static", "profile")
            os.makedirs(profile_dir, exist_ok=True)
            filename = f"avatar.{ext}"
            photo.save(os.path.join(profile_dir, filename))
            s["profile_photo"] = filename

    save_settings(s)
    flash("✅ 프로필 저장 완료!", "success")
    # 대시보드 헤더에서 연 모달이라 대시보드로 돌아가는 게 자연스러움
    return redirect(request.referrer or url_for("dashboard"))


@app.route("/settings", methods=["GET", "POST"])
def settings():
    s = load_settings()
    if request.method == "POST":
        action = request.form.get("action")

        # ── 복약 시간 삭제 ──
        if action == "delete_medicine":
            idx = int(request.form.get("idx", -1))
            if 0 <= idx < len(s["medicine_times"]):
                s["medicine_times"].pop(idx)
                save_settings(s)
                flash("✅ 복약 시간 삭제 완료!", "success")
            return redirect(url_for("settings"))

        # ── 복약 시간 — 폼에 있을 때만 업데이트 (기존 로직 그대로) ──
        hours   = request.form.getlist("med_hour")
        minutes = request.form.getlist("med_minute")
        if hours and minutes:
            new_times = [{"hour": int(h), "minute": int(m)}
                         for h, m in zip(hours, minutes) if h and m]
            if new_times:
                # 기존 복약 시간에 추가
                s["medicine_times"] = s.get("medicine_times", []) + new_times

        # 어느 섹션에서 온 요청인지 구분 (각 폼의 hidden input "section" 값)
        # 없으면(예: 복약 시간만 추가하는 요청) 다른 섹션은 건드리지 않음
        section = request.form.get("section", "")

        # ── 정각 알림 ──
        if section in ("chime", "all"):
            s["chime_enabled"] = "chime_enabled" in request.form

        # ── 음악 설정(얼굴 인식 시 자동 재생) ──
        if section in ("music", "all"):
            s["face_auto_music"] = "face_auto_music" in request.form

        # ── 슬라이드 속도 ──
        if section in ("slide", "all"):
            if "slide_interval" in request.form:
                s["slide_interval"] = int(request.form.get("slide_interval", 10))

        # ── 선다운 모드 ──
        if section in ("sundowner", "all"):
            s["sundowner_enabled"]    = "sundowner_enabled" in request.form
            s["sundowner_auto_music"] = "sundowner_auto_music" in request.form
            s["sundowner_hour"]       = int(request.form.get("sundowner_hour", 19))
            s["sundowner_minute"]     = int(request.form.get("sundowner_minute", 0))
            s["sundowner_end_hour"]   = int(request.form.get("sundowner_end_hour", 21))
            s["sundowner_end_minute"] = int(request.form.get("sundowner_end_minute", 0))

        save_settings(s)
        flash("✅ 설정 저장 완료!", "success")
        return redirect(url_for("settings"))

    return render_template("settings.html", settings=s)


# =============================================
# 메모리 페이지
# =============================================
@app.route("/memory", methods=["GET", "POST"])
def memory():
    mem = load_memory()
    if request.method == "POST":
        action = request.form.get("action")
        if action == "reset_all":
            save_memory({})
            flash("✅ 모든 기억 초기화!", "success")
        elif action == "reset_tag":
            tag = request.form.get("tag")
            if tag in mem:
                del mem[tag]
                save_memory(mem)
                flash(f"✅ '{tag}' 초기화!", "success")
        return redirect(url_for("memory"))
    sorted_mem = sorted(mem.items(), key=lambda x: -x[1])
    # 전체 합 대비 퍼센트 계산 — 절대 횟수가 아니라 상대적 비율로 보여줌
    total = sum(mem.values()) if mem else 0
    sorted_mem_with_pct = [
        (tag, count, round(count / total * 100, 1) if total > 0 else 0)
        for tag, count in sorted_mem
    ]
    viewing_stats = get_viewing_stats()
    speech_log    = load_speech_log()
    speech_summary = get_today_speech_summary()
    return render_template("memory.html", memory=sorted_mem_with_pct,
                           viewing=viewing_stats, speech=speech_log,
                           speech_summary=speech_summary)


@app.route("/media/photos/<path:filename>")
def serve_photo(filename):
    """사진 파일 서빙 — 썸네일 표시용"""
    return send_from_directory(PHOTOS_DIR, filename)

@app.route("/media/videos/<path:filename>")
def serve_video(filename):
    """영상 파일 서빙"""
    return send_from_directory(VIDEOS_DIR, filename)


def get_or_create_video_thumb(video_key):
    """
    video_key: "카테고리/파일명" 형식
    영상 첫 프레임을 추출해서 video_thumbs/ 폴더에 캐시해두고 경로 반환
    실패하면 터미널에 이유 출력하고 None 반환 (호출부에서 회색 박스로 폴백 처리)

    ⚠️ 윈도우에서 cv2는 한글(유니코드) 경로를 다루는 데 취약함:
      - cv2.VideoCapture: 한글 경로의 영상을 못 여는 경우가 있어서
        → 열기 전에 영문 이름의 임시 파일로 복사해서 그걸 대신 엶
      - cv2.imwrite: 한글 경로로 저장하면 에러 없이 "조용히 실패"하는 경우가 있어서
        → 캐시 파일명 자체를 해시값(순수 영문+숫자)으로 만들어서 회피
    """
    import tempfile
    import hashlib

    os.makedirs(THUMB_DIR, exist_ok=True)
    safe_name = hashlib.md5(video_key.encode("utf-8")).hexdigest() + ".jpg"
    thumb_path = os.path.join(THUMB_DIR, safe_name)

    if os.path.exists(thumb_path):
        return thumb_path

    video_path = os.path.join(VIDEOS_DIR, video_key)
    if not os.path.exists(video_path):
        print(f"⚠️ 영상 썸네일 실패 - 파일 없음: {video_path}")
        return None

    ext = video_path.rsplit(".", 1)[-1]
    tmp_path = None
    try:
        # 한글 경로 우회용 임시 파일(영문 경로)로 복사
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
            tmp_path = tmp.name
        shutil.copyfile(video_path, tmp_path)

        cap = cv2.VideoCapture(tmp_path)
        success, frame = cap.read()
        cap.release()
        if not success:
            print(f"⚠️ 영상 썸네일 실패 - 프레임을 못 읽음(코덱 문제일 수 있음): {video_path}")
            return None
        write_ok = cv2.imwrite(thumb_path, frame)
        if not write_ok:
            print(f"⚠️ 영상 썸네일 실패 - 썸네일 파일 저장(imwrite) 실패: {thumb_path}")
            return None
        return thumb_path
    except Exception as e:
        print(f"⚠️ 영상 썸네일 실패 - 예외 발생: {video_path} / {e}")
        return None
    finally:
        # 임시 복사본은 썸네일 추출 후 바로 삭제 (계속 남으면 디스크만 차지함)
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@app.route("/media/video-thumb/<path:filename>")
def video_thumbnail(filename):
    """
    영상의 첫 프레임을 썸네일 이미지로 반환
    - "많이 본 사진 주제" 등에서 최다 반응 콘텐츠가 사진이 아니라 영상일 때 사용
    - 실제 추출/캐싱은 get_or_create_video_thumb()에서 처리
    """
    thumb_path = get_or_create_video_thumb(filename)
    if not thumb_path:
        return "", 404
    directory, name = os.path.split(thumb_path)

    directory, name = os.path.split(thumb_path)
    return send_from_directory(directory, name, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
