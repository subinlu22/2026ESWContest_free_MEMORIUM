import pygame
import os
import sys
import threading
from gtts import gTTS
import tempfile
import datetime
import random
import re
import cv2

# [최적화] DeepFace 내부 TensorFlow가 CPU 코어를 전부 차지하지 않도록 미리 제한
# → 코어 몇 개는 항상 남겨둬서, 음악 재생/화면 그리기가 순간적으로 밀리는 것을 막음
# (반드시 deepface를 import하기 "전"에 설정해야 실제로 적용됨)
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

from deepface import DeepFace

# =============================================
# 메모리움 (Memorium)
# 치매 노인을 위한 AI 스마트 액자
# =============================================
# 기능 목록:
#   - 사진 + 영상 랜덤 슬라이드쇼 (슬라이드 전환 애니메이션)
#   - 배경음악 랜덤 재생 (M키 토글)
#   - TTS 안내 (시보, 복약 알림, 얼굴 인식 인사)
#   - 얼굴 감지 → 화면 테두리 밝아짐 + 음악 켜짐
#   - 표정 인식 → 웃음 2회 감지 시 선호 태그 미디어 2장 우선 재생
# =============================================


# =============================================
# ■ 경로 설정
# =============================================
# BASE_DIR: 이 스크립트 파일이 있는 폴더를 자동으로 잡음 (윈도우/리눅스 둘 다 동작)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PHOTOS_DIR = os.path.join(BASE_DIR, "photos")   # 사진 폴더
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")   # 영상 폴더
MUSIC_DIR   = os.path.join(BASE_DIR, "music")    # 음악 폴더
MUSIC_CALM_DIR = os.path.join(BASE_DIR, "music_calm")  # 썬다운 모드용 차분한 음악 폴더
MEMORY_FILE = os.path.join(BASE_DIR, "memory.json")  # 태그 가중치 저장 파일
TAGS_FILE      = os.path.join(BASE_DIR, "tags.json")      # 파일별 태그 저장 파일
SETTINGS_FILE  = os.path.join(BASE_DIR, "settings.json")  # 설정 저장 파일
FONT_PATH      = os.path.join(BASE_DIR, "fonts", "Cafe24Ssurround-v2.0.ttf")  # 날짜 표시용 폰트
VIEWING_LOG_FILE = os.path.join(BASE_DIR, "viewing_log.json")  # 액자 응시 횟수/시간 기록 파일
SPEECH_LOG_FILE  = os.path.join(BASE_DIR, "speech_log.json")   # 어르신 말씀 텍스트 기록 파일
SPEECH_ARCHIVE_FILE = os.path.join(BASE_DIR, "speech_log_archive.json")  # 오래된 말씀 기록 보관용 (90일 넘은 것들)


# =============================================
# ■ 화면 설정
# =============================================
SCREEN_WIDTH  = 900   # 시작은 창모드 크기 (F키로 전체화면 전환 시 실제 모니터 해상도로 바뀜)
SCREEN_HEIGHT = 600
WINDOWED_WIDTH  = 900   # F키로 창모드 복귀할 때 돌아갈 기본 크기
WINDOWED_HEIGHT = 600
BG_COLOR      = (0, 0, 0)   # 배경 색상 (검정)


# =============================================
# ■ 슬라이드쇼 설정
# =============================================
# SLIDE_INTERVAL은 settings.json에서 읽어옴 (기본 10초)
SLIDE_SPEED    = 30      # 슬라이드 애니메이션 속도 (픽셀/프레임)


# =============================================
# ■ 표정 인식 설정
# =============================================
ANALYZE_EVERY  = 10    # (미사용 — 표정 분석이 별도 스레드로 분리되며 프레임 카운트 방식은 폐기됨)
FACE_ANALYSIS_INTERVAL = 0.5   # 표정 분석 스레드가 한 번 분석 후 쉬는 시간(초). 낮을수록 자주 분석하지만 CPU 더 씀
HAPPY_THRESHOLD = 90   # 웃음으로 인정할 happy 확률 (%) — 낮추면 더 잘 감지 (오탐 많아서 80→90 상향)
SMILE_NEEDED   = 2     # 선호 태그 설정에 필요한 웃음 감지 횟수


# =============================================
# ■ 선호 태그 설정
# =============================================
PREFER_RATIO = 0.7   # 선호 태그 미디어 노출 비율 (0.7 = 70%)
PREFER_MAX   = 2     # 선호 태그 미디어 연속 재생 횟수


# =============================================
# ■ 얼굴 감지 설정
# =============================================
FACE_CONFIDENCE  = 0.5   # 얼굴로 인정할 최소 신뢰도
NO_FACE_THRESHOLD = 5    # 연속 N번 얼굴 못 찾아야 사라진 걸로 처리


# =============================================
# ■ 테두리 설정
# =============================================
BORDER_MAX       = 200   # 얼굴 없을 때 테두리 어두움 (0~255)
BORDER_MIN       = 0     # 얼굴 있을 때 테두리 (0 = 없음)
BORDER_THICKNESS = 80    # 테두리 최대 두께 (픽셀)
ALPHA_STEP       = 5     # 테두리 밝기 변화 속도


# =============================================
# ■ 알림 설정 (settings.json에서 읽어옴)
# =============================================
import json as _json

def load_settings():
    """settings.json에서 설정 불러오기. 없으면 기본값 사용"""
    settings_path = os.path.join(os.path.dirname(__file__), "settings.json")
    default = {
        "medicine_times": [{"hour": 14, "minute": 0}],
        "chime_enabled": True,
        "sundowner_enabled": True,
        "sundowner_hour": 19
    }
    if os.path.exists(settings_path):
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                return _json.load(f)
        except Exception:
            pass
    return default

_settings      = load_settings()
MEDICINE_TIMES = [(t["hour"], t["minute"]) for t in _settings.get("medicine_times", [{"hour": 14, "minute": 0}])]
CHIME_ENABLED  = _settings.get("chime_enabled", True)
CHIME_HOURS    = list(range(7, 22)) if CHIME_ENABLED else []
SUNDOWNER_ENABLED = _settings.get("sundowner_enabled", True)
SUNDOWNER_HOUR    = _settings.get("sundowner_hour", 19)
SUNDOWNER_MINUTE  = _settings.get("sundowner_minute", 0)
SUNDOWNER_END_HOUR   = _settings.get("sundowner_end_hour", 21)
SUNDOWNER_END_MINUTE = _settings.get("sundowner_end_minute", 0)
SUNDOWNER_AUTO_MUSIC = _settings.get("sundowner_auto_music", True)  # 썬다운 시 자동 음악 켤지
FACE_AUTO_MUSIC      = _settings.get("face_auto_music", True)       # 평소 얼굴 인식 시 자동 음악 켤지
SLIDE_INTERVAL    = _settings.get("slide_interval", 10) * 1000  # 초 → 밀리초 변환
print(f"설정 불러옴 — 복약: {MEDICINE_TIMES}, 시보: {CHIME_ENABLED}, 슬라이드: {SLIDE_INTERVAL//1000}초")


# =============================================
# ■ pygame 이벤트
# =============================================
MUSIC_END_EVENT = pygame.USEREVENT + 1   # 곡 끝났을 때 발생하는 커스텀 이벤트

# 전역 음악 상태
music_on = True

# TTS 우선순위 (숫자 낮을수록 높은 우선순위)
# 1: 복약 알림, 2: 시보, 3: 인사
current_tts_priority = 99   # 현재 재생 중인 TTS 우선순위 (99 = 없음)

# TTS 우선순위 (숫자 낮을수록 높은 우선순위)
# 1: 복약 알림, 2: 시보, 3: 인사
TTS_PRIORITY = {"복약": 1, "시보": 2, "인사": 3}
current_tts_priority = 99   # 현재 재생 중인 TTS 우선순위 (99 = 없음)


# =============================================
# TTS 함수
# =============================================
def speak(text, priority=3):
    """
    텍스트를 한국어 음성으로 출력
    priority: 1=복약(최고), 2=시보, 3=인사(최저)
    현재 재생 중인 TTS보다 우선순위 낮으면 무시
    """
    global current_tts_priority
    if priority > current_tts_priority:
        return

    def _speak():
        global current_tts_priority
        try:
            current_tts_priority = priority
            tts = gTTS(text=text, lang="ko")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                temp_path = f.name
            tts.save(temp_path)
            pygame.mixer.music.set_volume(0.1)   # TTS 중 음악 볼륨 낮춤
            sound = pygame.mixer.Sound(temp_path)
            sound.play()
            while pygame.mixer.get_busy():
                pygame.time.Clock().tick(10)
            if music_on:
                pygame.mixer.music.set_volume(0.5)  # TTS 끝나면 볼륨 복구
            os.remove(temp_path)
        except Exception as e:
            print(f"TTS 오류: {e}")
        finally:
            current_tts_priority = 99  # 재생 끝나면 초기화
    threading.Thread(target=_speak, daemon=True).start()



# =============================================
# 태그 가중치 기억 시스템
# =============================================
def load_memory():
    """
    memory.json에서 태그 가중치 불러오기
    예: {"꽃": 5, "동물": 2, "풍경": 0}
    파일 없으면 빈 딕셔너리 반환
    """
    import json
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_memory(memory):
    """태그 가중치를 memory.json에 저장"""
    import json
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


def add_memory(memory, tag):
    """웃음 감지 시 해당 태그 가중치 +1 후 저장"""
    memory[tag] = memory.get(tag, 0) + 1
    save_memory(memory)
    print(f"기억 업데이트: {tag} = {memory[tag]}회")
    return memory


def load_viewing_log():
    """viewing_log.json에서 응시 기록 목록 불러오기. 파일 없으면 빈 목록 반환"""
    import json
    if os.path.exists(VIEWING_LOG_FILE):
        try:
            with open(VIEWING_LOG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def add_viewing_record(start_time, media_path=None):
    """
    얼굴이 사라지는 시점에 호출 — 응시 시작~종료 시간을 계산해서 기록 1줄 추가
    start_time: 얼굴이 처음 감지된 시각 (datetime 객체)
    media_path: [신규] 그 시점에 보여주고 있던 사진/영상의 전체 경로 (없으면 기록 안 함)
                → 웹 대시보드 "최근 활동"에서 실제 사진/영상 썸네일을 보여주는 데 사용
    너무 짧은 응시(3초 미만)는 오감지일 수 있어 기록하지 않음
    """
    import json
    end_time = datetime.datetime.now()
    duration_seconds = (end_time - start_time).total_seconds()

    if duration_seconds < 3:
        return  # 너무 짧으면 오감지로 보고 기록 안 함

    record = {
        "date": start_time.strftime("%Y-%m-%d"),
        "start": start_time.strftime("%H:%M:%S"),
        "duration_minutes": round(duration_seconds / 60, 1)
    }

    # [신규] media_path가 있으면 "카테고리/파일명" 형식의 키로 변환해서 같이 저장
    # (get_tags()에서 쓰는 것과 동일한 키 형식 — 영상이면 앞에 "v:" 붙임)
    if media_path:
        filename = os.path.basename(media_path)
        parent   = os.path.basename(os.path.dirname(media_path))
        if is_video(media_path):
            record["media_key"] = f"v:{parent}/{filename}"
        else:
            record["media_key"] = f"{parent}/{filename}"

    log = load_viewing_log()
    log.append(record)
    with open(VIEWING_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"응시 기록 저장: {start_time.strftime('%H:%M:%S')}부터 {round(duration_seconds/60, 1)}분")


def save_speech_log(text):
    """
    STT로 변환된 텍스트를 speech_log.json에 저장
    날짜/시각과 함께 기록 — 보호자가 웹에서 확인용
    """
    import json
    log = []
    if os.path.exists(SPEECH_LOG_FILE):
        try:
            with open(SPEECH_LOG_FILE, "r", encoding="utf-8") as f:
                log = json.load(f)
        except Exception:
            pass
    now = datetime.datetime.now()
    log.append({
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M"),
        "text": text
    })
    with open(SPEECH_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"말씀 기록: {text}")


def archive_old_speech(threshold_days=90):
    """
    speech_log.json에 threshold_days일(기본 90일) 넘은 기록이 있으면
    speech_log_archive.json으로 옮기고, speech_log.json은 최근 기록만 남김.

    이유: save_speech_log()는 말씀 한 마디 저장할 때마다 파일 전체를
          읽고 다시 쓰는 방식이라, 기록이 몇 년치 쌓이면 저장할 때마다
          조금씩 느려질 수 있음. 오래된 기록을 별도 파일로 분리해서
          메인 파일을 가볍게 유지하는 용도.
    프로그램 시작할 때 한 번만 실행됨 (매번 실행할 필요는 없음).
    """
    import json
    import datetime as dt

    if not os.path.exists(SPEECH_LOG_FILE):
        return

    try:
        with open(SPEECH_LOG_FILE, "r", encoding="utf-8") as f:
            log = json.load(f)
    except Exception:
        return  # 파일 읽기 실패하면 안전하게 그냥 건너뜀

    cutoff = dt.datetime.now() - dt.timedelta(days=threshold_days)
    recent, old = [], []
    for r in log:
        try:
            record_date = dt.datetime.strptime(r["date"], "%Y-%m-%d")
        except Exception:
            recent.append(r)  # 날짜 파싱 실패하면 안전하게 최근으로 취급 (삭제 방지)
            continue
        if record_date >= cutoff:
            recent.append(r)
        else:
            old.append(r)

    if not old:
        return  # 옮길 만큼 오래된 기록이 없으면 그냥 끝

    archive = []
    if os.path.exists(SPEECH_ARCHIVE_FILE):
        try:
            with open(SPEECH_ARCHIVE_FILE, "r", encoding="utf-8") as f:
                archive = json.load(f)
        except Exception:
            pass
    archive.extend(old)

    with open(SPEECH_ARCHIVE_FILE, "w", encoding="utf-8") as f:
        json.dump(archive, f, ensure_ascii=False, indent=2)
    with open(SPEECH_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(recent, f, ensure_ascii=False, indent=2)

    print(f"오래된 말씀 기록 {len(old)}개 보관 완료 → speech_log_archive.json (남은 최근 기록 {len(recent)}개)")


def start_stt_listener(face_detected_ref):
    """
    백그라운드 스레드로 실행되는 STT 리스너
    face_detected_ref: [bool] 형태의 리스트 — 얼굴 감지 여부를 메인 루프와 공유
    얼굴이 인식된 동안에만 마이크를 듣고, 말소리 감지 시 텍스트로 변환해서 저장
    리스트로 감싸는 이유: 스레드 간에 변수를 공유할 때 리스트 안의 값을 바꾸면
                          메인 루프에서도 변경이 반영되기 때문 (아까 SLIDE_INTERVAL 때랑 같은 방식)
    """
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        mic = sr.Microphone()

        # 주변 소음 수준 자동 보정 (처음 1초)
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
        print("STT 리스너 시작 — 얼굴 인식 시 자동 감지")

        while True:
            if not face_detected_ref[0]:
                # 얼굴 없으면 잠깐 쉬고 다시 확인
                import time
                time.sleep(0.5)
                continue

            # 얼굴 감지 중 — 마이크 켜고 말소리 대기
            try:
                with mic as source:
                    # timeout=3: 3초 안에 말 안 하면 다시 루프
                    # phrase_time_limit=10: 한 번에 최대 10초까지 인식
                    audio = recognizer.listen(source, timeout=3, phrase_time_limit=10)
                text = recognizer.recognize_google(audio, language="ko-KR")
                if text.strip():
                    save_speech_log(text.strip())
            except sr.WaitTimeoutError:
                print("[진단] 3초 안에 마이크가 소리를 못 들음 (타임아웃)")
            except sr.UnknownValueError:
                print("[진단] 소리는 들었는데 텍스트 변환 실패 (인식 안 됨)")
            except Exception as e:
                print(f"STT 오류: {e}")

    except Exception as e:
        print(f"STT 리스너 시작 실패: {e}")


def start_face_analysis_thread(cap, state):
    """
    표정 분석 전용 백그라운드 스레드
    - 무거운 DeepFace.analyze()를 메인 루프 밖에서 독립적으로 실행
    - 분석이 끝날 때마다 결과(얼굴 신뢰도, 웃음 점수)를 공유 딕셔너리(state)에 기록
    - 메인 루프는 이 값을 읽기만 하므로, 분석이 아무리 오래 걸려도
      화면 그리기·음악 재생 타이밍에는 전혀 영향을 안 줌
      (STT 리스너를 별도 스레드로 돌리던 것과 동일한 방식)

    state: {"confidence": 0, "happy": 0, "new": False, "lock": threading.Lock()}
           - "new"는 메인 루프가 아직 안 읽어간 새 결과가 있는지 표시하는 플래그
           - lock으로 감싸는 이유: 스레드 두 개(이 스레드 / 메인 루프)가
             동시에 값을 읽고 쓸 수 있어서, 값이 반쯤 바뀐 상태로 읽히는
             것을 막기 위함
    """
    import time
    while True:
        ret_f, frame_f = cap.read()
        if ret_f:
            try:
                result      = DeepFace.analyze(frame_f, actions=["emotion"],
                                               enforce_detection=False, silent=True)
                confidence  = result[0].get("face_confidence", 0)
                happy_score = result[0]["emotion"].get("happy", 0)
                with state["lock"]:
                    state["confidence"] = confidence
                    state["happy"]      = happy_score
                    state["new"]        = True
            except Exception as e:
                print(f"DeepFace 에러: {e}")
        time.sleep(FACE_ANALYSIS_INTERVAL)   # 다음 분석까지 잠깐 쉼 (CPU 여유 주기용 안전장치)


# =============================================
# 파일 유틸 함수
# =============================================
def is_video(path):
    """파일이 영상인지 확인"""
    return path.lower().endswith((".mp4", ".avi", ".mov"))


def load_tags():
    """tags.json에서 파일별 태그 목록 불러오기
    예: {"꽃1.jpg": ["꽃", "자연"], "가족1.jpg": ["가족", "손녀"]}
    """
    import json
    if os.path.exists(TAGS_FILE):
        try:
            with open(TAGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


# 전역 태그 딕셔너리 (실행 시 한 번 로드)
_tags_cache = {}


def get_tags(path):
    """
    파일에 연결된 태그 목록 반환 (중복 태그 지원)
    사진 키: "카테고리/파일명"
    영상 키: "v:카테고리/파일명"
    없으면 폴더명을 태그로 사용
    """
    filename = os.path.basename(path)
    parent   = os.path.basename(os.path.dirname(path))

    # 영상 카테고리 키로 찾기 (v:카테고리/파일명)
    video_cat_key = f"v:{parent}/{filename}"
    if video_cat_key in _tags_cache:
        return _tags_cache[video_cat_key]

    # 사진 카테고리 키로 찾기 (카테고리/파일명)
    cat_key = f"{parent}/{filename}"
    if cat_key in _tags_cache:
        return _tags_cache[cat_key]

    # 파일명 키로 찾기
    if filename in _tags_cache:
        return _tags_cache[filename]

    # 폴더명을 태그로 (하위 폴더에 있으면 폴더명 = 카테고리)
    base_dirs = [PHOTOS_DIR.rstrip(os.sep), VIDEOS_DIR.rstrip(os.sep)]
    parent_path = os.path.dirname(path)
    if parent_path not in base_dirs:
        return [parent]

    # 파일명 기반 폴백
    name = os.path.splitext(filename)[0].replace("영상", "")
    if "_" in name:
        return [name.split("_")[0]]
    match = re.match(r'([가-힣]+)', name)
    return [match.group(1)] if match else ["기타"]


def get_tag(path):
    """
    대표 태그 하나 반환 (기존 호환용)
    중복 태그가 있으면 첫 번째 태그 반환
    """
    tags = get_tags(path)
    return tags[0] if tags else "기타"


def load_images(folder):
    """
    사진 폴더에서 jpg/png 파일 목록 반환
    하위 카테고리 폴더도 탐색 (photos/풍경/바다.jpg 등)
    """
    extensions = (".jpg", ".jpeg", ".png")
    images = []
    for entry in os.listdir(folder):
        entry_path = os.path.join(folder, entry)
        if os.path.isdir(entry_path):
            # 카테고리 하위 폴더 탐색
            for f in os.listdir(entry_path):
                if f.lower().endswith(extensions):
                    images.append(os.path.join(entry_path, f))
        elif entry.lower().endswith(extensions):
            images.append(entry_path)
    return images


def load_videos(folder):
    """
    영상 폴더에서 mp4/avi/mov 파일 목록 반환
    하위 카테고리 폴더도 탐색 (videos/풍경/바다.mp4 등)
    """
    if not os.path.exists(folder):
        return []
    extensions = (".mp4", ".avi", ".mov")
    videos = []
    for entry in os.listdir(folder):
        entry_path = os.path.join(folder, entry)
        if os.path.isdir(entry_path):
            for f in os.listdir(entry_path):
                if f.lower().endswith(extensions):
                    videos.append(os.path.join(entry_path, f))
        elif entry.lower().endswith(extensions):
            videos.append(entry_path)
    return videos


def load_music(folder):
    """음악 폴더에서 mp3/wav 파일 목록 랜덤 순서로 반환"""
    extensions = (".mp3", ".wav")
    music = [os.path.join(folder, f) for f in os.listdir(folder)
             if f.lower().endswith(extensions)]
    random.shuffle(music)
    return music


def fit_image(surface, screen_w, screen_h):
    """
    이미지를 비율 유지하면서 화면을 꽉 채우게 리사이즈
    (화면 비율과 안 맞아서 남는 부분은 중앙 기준으로 잘라냄 — 여백/레터박스 없이 꽉 채워 보임)
    """
    img_w, img_h = surface.get_size()
    ratio = max(screen_w / img_w, screen_h / img_h)   # 화면을 꽉 채우도록 큰 비율 선택 (기존엔 min이라 안쪽에 맞춰져서 여백 생겼음)
    new_w, new_h = int(img_w * ratio) + 1, int(img_h * ratio) + 1  # 반올림 오차로 화면보다 살짝 작아지는 것 방지
    surface = surface.convert_alpha()  # 포맷 통일
    scaled = pygame.transform.smoothscale(surface, (new_w, new_h))

    crop_x = max((new_w - screen_w) // 2, 0)
    crop_y = max((new_h - screen_h) // 2, 0)
    return scaled.subsurface((crop_x, crop_y, screen_w, screen_h)).copy()   # 화면 벗어나는 부분 중앙 기준으로 잘라냄


# =============================================
# 다음 미디어 선택
# =============================================
def pick_next(all_media, current_index, prefer_tag=None, force=False, memory=None, sundowner_active=False):
    """
    다음에 재생할 미디어 인덱스 선택
    - prefer_tag + force=True: 해당 태그 미디어 100% 선택 (선호 태그 2장 보장용)
    - prefer_tag + force=False: PREFER_RATIO 확률로 해당 태그 선택
    - prefer_tag 없으면 가중치(memory) 기반 랜덤 선택
    - sundowner_active=True면 '인물(가족)' 태그 미디어가 30% 랜덤 구간에서 우선 선택됨
      (썬다운 증후군 대응 — 익숙한 가족 얼굴이 안정감을 주는 효과. 좋아하는 태그 우선순위는 그대로 유지)
    """
    if prefer_tag:
        # 중복 태그 지원 — prefer_tag가 태그 목록에 포함되면 선호
        preferred = [i for i, p in enumerate(all_media)
                     if prefer_tag in get_tags(p) and i != current_index]
        others    = [i for i, p in enumerate(all_media)
                     if prefer_tag not in get_tags(p) and i != current_index]
        if preferred and (force or random.random() < PREFER_RATIO):
            return random.choice(preferred)   # 선호 태그 100% or 70%
        elif others:
            return random.choice(others)
        elif preferred:
            return random.choice(preferred)

    # 가중치 기반 랜덤 선택
    candidates = [i for i in range(len(all_media)) if i != current_index]
    if not candidates:
        return current_index

    # 30% 확률 구간 — 평소엔 완전 랜덤, 썬다운 시간대엔 '인물(가족)' 우선 랜덤
    # (익숙한 가족 얼굴이 사이사이 섞여 나오면서 안정감을 주되, 좋아하는 태그 우선순위는 그대로 유지)
    if random.random() < 0.3 or not memory:
        if sundowner_active:
            family_candidates = [i for i in candidates if "인물" in get_tags(all_media[i])]
            if family_candidates:
                return random.choice(family_candidates)
        return random.choice(candidates)

    # 70% 확률로 가중치 기반 선택 (좋아하는 태그 우선 — 평소와 동일한 로직)
    now = datetime.datetime.now().timestamp()

    # 전체 태그 누적 횟수의 합 — 비율 계산의 기준값
    # (절대 숫자가 아무리 커져도, 비율로 계산하면 한 태그가 무한정 압도하지 않음.
    #  기억은 평생 그대로 보존되면서도, 다른 취향과의 상대적 균형이 유지됨)
    total_memory = sum(memory.values()) if memory else 0

    weights = []
    for i in candidates:
        path = all_media[i]
        tags = get_tags(path)
        if total_memory > 0:
            # 이 미디어가 가진 태그들의 누적 비율 합 (0~1 사이, 여러 태그면 더 커질 수 있음)
            tag_ratio = sum(memory.get(t, 0) for t in tags) / total_memory
        else:
            tag_ratio = 0
        # 비율에 배율(10)을 곱해 가중치로 변환 — 기본 가중치 1에 더함
        w = 1 + (tag_ratio * 10)
        # 최근 7일 이내 파일이면 가중치 +2 (새 사진 우선 노출)
        try:
            file_age_days = (now - os.path.getmtime(path)) / 86400
            if file_age_days <= 7:
                w += 2
        except Exception:
            pass
        weights.append(w)
    return random.choices(candidates, weights=weights, k=1)[0]


# =============================================
# 음악 재생
# =============================================
def play_music(music_paths, index, fade_in_ms=0, volume=0.5):
    """
    지정한 인덱스의 곡 재생 — 끝나면 MUSIC_END_EVENT 발생
    fade_in_ms: 0이면 즉시 재생(기존과 동일), 숫자를 주면 그 시간(밀리초)에 걸쳐
                볼륨이 서서히 커지면서 재생 시작 (예: 3000 = 3초에 걸쳐 부드럽게 시작)
    volume: 도달할 목표 볼륨 (기본 0.5, 평소와 동일)
    """
    pygame.mixer.music.load(music_paths[index])
    pygame.mixer.music.set_volume(volume)
    pygame.mixer.music.set_endevent(MUSIC_END_EVENT)
    pygame.mixer.music.play(fade_ms=fade_in_ms)
    print(f"배경음악: {os.path.basename(music_paths[index])}")


# =============================================
# 영상 프레임 그리기
# =============================================
def play_video_frame(cap_v, screen, border_alpha):
    """
    영상에서 프레임 하나 읽어서 화면에 그리기
    - 비율 유지하면서 화면을 꽉 채움 (화면 비율과 안 맞는 부분은 중앙 기준으로 잘라냄)
    - 끝나면 True 반환

    [최적화]
    1) 원래는 '원본 크기 그대로 색상변환(BGR→RGB) → 리사이즈' 순서였는데,
       리사이즈를 먼저 해서 화면 크기만큼 작아진 다음 색상변환하도록 순서를 바꿈
       (처리해야 할 픽셀 수 자체가 줄어드니 매 프레임 연산량이 줄어듦)
    2) surfarray.make_surface + transpose 조합 대신 pygame.image.frombuffer 사용
       (배열을 뒤집어서 복사하던 과정 없이 바로 화면에 그릴 수 있는 형태로 변환돼서 더 빠름)
    3) 화면 꽉 채우기: min 대신 max 비율로 확대한 뒤, 화면 크기만큼 중앙에서 잘라냄
       (자르는 것도 색상변환 '전'에 배열 슬라이싱으로 처리해서 연산량을 더 줄임)
    """
    ret, frame = cap_v.read()
    if not ret:
        return True   # 영상 끝

    h, w = frame.shape[:2]
    ratio = max(SCREEN_WIDTH / w, SCREEN_HEIGHT / h)   # 화면을 꽉 채우도록 큰 비율 선택 (기존엔 min이라 여백 생겼음)
    new_w, new_h = int(w * ratio) + 1, int(h * ratio) + 1  # 반올림 오차로 화면보다 살짝 작아지는 것 방지

    frame_small = cv2.resize(frame, (new_w, new_h))          # 리사이즈 먼저 (처리할 픽셀 수 감소)

    crop_x = max((new_w - SCREEN_WIDTH) // 2, 0)
    crop_y = max((new_h - SCREEN_HEIGHT) // 2, 0)
    frame_cropped = frame_small[crop_y:crop_y + SCREEN_HEIGHT, crop_x:crop_x + SCREEN_WIDTH].copy()  # 화면 벗어나는 부분 중앙 기준으로 잘라냄

    frame_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)  # 그 다음 색상변환 (화면 크기만큼만 대상)

    surf = pygame.image.frombuffer(frame_rgb.tobytes(), (SCREEN_WIDTH, SCREEN_HEIGHT), "RGB")
    screen.fill(BG_COLOR)
    screen.blit(surf, (0, 0))
    draw_border(screen, border_alpha)
    draw_date(screen, get_date_text(), 255)   # 영상 재생 중에도 날짜 항상 표시
    pygame.display.flip()
    return False


# =============================================
# 테두리 오버레이
# =============================================
_vignette_cache = {}   # {(screen_w, screen_h): {"mask": 정규화된 비네팅 모양(0~1), "surface": 재사용할 Surface, "last_alpha": 마지막으로 그렸던 alpha값}}

def draw_border(screen, alpha):
    """
    비네팅 효과 — 위/아래만 어두워지는 그라데이션 (좌우는 그대로 둠)
    세로 사진이든 가로 사진이든 동일하게 자연스럽게 보이도록 세로 방향만 계산
    - alpha=0: 비네팅 없음 (얼굴 감지 시)
    - alpha=200: 비네팅 강함 (얼굴 없을 때)

    [최적화] 원래는 매 프레임마다 화면 전체 크기로 numpy 배열/Surface를 새로 만들었는데,
    비네팅 '모양' 자체는 화면 크기가 안 바뀌면 항상 똑같음. 그래서 모양 계산은
    화면 크기별로 딱 1번만 하고 캐싱해뒀다가 재사용함. alpha 값도 이전 프레임과
    같으면(자주 그럼) 다시 그릴 필요 없이 이전에 만든 Surface 그대로 씀.
    전체화면처럼 해상도가 커질수록 이 캐싱 효과가 훨씬 커짐 (연산량이 화면 크기에 비례하기 때문).
    """
    if alpha <= 0:
        return

    import numpy as np

    key = (SCREEN_WIDTH, SCREEN_HEIGHT)
    cache = _vignette_cache.get(key)

    if cache is None:
        # 화면 크기가 처음 나왔을 때(또는 F키로 바뀌었을 때)만 모양 자체를 계산
        cy = SCREEN_HEIGHT / 2
        y_idx = np.arange(SCREEN_HEIGHT).reshape(-1, 1)
        dist = np.abs(y_idx - cy) / cy   # 위/아래로 갈수록 1에 가까워짐
        mask = np.clip((dist - 0.4) / 0.6, 0, 1) ** 1.5   # (H, 1), 0~1 정규화된 비네팅 모양
        mask = np.broadcast_to(mask, (SCREEN_HEIGHT, SCREEN_WIDTH)).copy()  # (H, W)로 가로 방향 반복

        surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        cache = {"mask": mask, "surface": surface, "last_alpha": None}
        _vignette_cache[key] = cache

    # alpha 값이 이전 프레임과 같으면 다시 계산 안 하고 캐시된 Surface 그대로 재사용
    if cache["last_alpha"] != alpha:
        vignette_alpha = (cache["mask"] * alpha).astype(np.uint8)
        pygame.surfarray.pixels_alpha(cache["surface"])[:] = vignette_alpha.T  # 기존 Surface 재활용 (새로 안 만듦)
        cache["last_alpha"] = alpha

    screen.blit(cache["surface"], (0, 0))


# =============================================
# 시각 안내 텍스트 생성
# =============================================
def get_time_text():
    """현재 시각을 자연스러운 한국어 문장으로 변환"""
    now = datetime.datetime.now()
    h, m = now.hour, now.minute
    if h < 12:    period, hh = "오전", h
    elif h == 12: period, hh = "낮", 12
    else:         period, hh = "오후", h - 12
    return f"지금은 {period} {hh}시입니다" if m == 0 else f"지금은 {period} {hh}시 {m}분입니다"


def get_date_text():
    """화면에 표시할 오늘 날짜/요일 텍스트 생성 (예: 6월 30일 화요일)"""
    weekdays = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
    now = datetime.datetime.now()
    return f"{now.month}월 {now.day}일 {weekdays[now.weekday()]}"


def get_greeting_text(hour=None):
    """
    얼굴 감지 시 시간대별로 다르게 나갈 인사 멘트 생성
    - 아침(~11시): 좋은 아침이에요
    - 점심(11~14시): 점심은 드셨어요?
    - 오후(14~18시): 안녕하세요
    - 저녁(18시~): 오늘 하루도 수고하셨어요

    hour: 테스트용 — 숫자를 직접 넣으면 그 시각인 척 멘트를 확인할 수 있음
          평소 실행할 땐 아무것도 안 넣으면 자동으로 지금 시각을 씀
    """
    if hour is None:
        hour = datetime.datetime.now().hour
    h = hour
    if h < 11:
        return "좋은 아침이에요"
    elif h < 14:
        return "점심은 드셨어요?"
    elif h < 18:
        return "안녕하세요"
    else:
        return "오늘 하루도 수고하셨어요"


_date_cache = {"font": None, "last_text": None, "last_alpha": None, "surface": None}

def draw_date(screen, text, alpha):
    """
    화면 우하단에 날짜/요일 표시
    alpha: 0~255 투명도 (현재는 항상 255로 고정 표시)

    [최적화] 원래는 매 프레임마다 폰트 파일을 디스크에서 새로 읽어들이고
    텍스트도 매번 새로 렌더링했음 (특히 폰트 파일 디스크 로드가 큰 부담이었음).
    폰트는 프로그램 켜져있는 동안 딱 1번만 로드해서 재사용하고, 텍스트 내용과
    alpha 값이 이전 프레임과 똑같으면(날짜는 하루 종일 안 바뀌므로 거의 항상 그럼)
    다시 렌더링하지 않고 이전에 만든 결과를 그대로 재사용함.
    """
    if alpha <= 0:
        return

    if _date_cache["font"] is None:
        _date_cache["font"] = pygame.font.Font(FONT_PATH, 28)   # 폰트는 최초 1번만 디스크에서 로드

    if _date_cache["last_text"] != text or _date_cache["last_alpha"] != alpha:
        surf = _date_cache["font"].render(text, True, (255, 255, 255))
        surf.set_alpha(alpha)
        _date_cache["surface"]    = surf
        _date_cache["last_text"]  = text
        _date_cache["last_alpha"] = alpha
    else:
        surf = _date_cache["surface"]   # 안 바뀌었으면 새로 안 만들고 재사용

    text_w = surf.get_width()
    text_h = surf.get_height()
    x = (SCREEN_WIDTH - text_w) // 2  # 가로 중앙 정렬
    y = SCREEN_HEIGHT - text_h - 20   # 화면 하단에서 20px 띄운 위치
    screen.blit(surf, (x, y))         # 하단 중앙에 배치


# =============================================
# 메인 함수
# =============================================
def main():
    global music_on
    global SCREEN_WIDTH, SCREEN_HEIGHT   # [전체화면] fit_image 등 다른 함수도 이 값을 그대로 참조하므로 전역값 자체를 갱신

    pygame.init()
    pygame.mixer.init()

    # [전체화면] 창을 만들기 '전'에 진짜 모니터 해상도를 미리 저장해둠
    # (창을 만든 '후'에 Info()를 부르면 라즈베리파이 환경에서 방금 만든 창 크기가
    #  그대로 돌아오는 경우가 있어서, 반드시 set_mode() 호출 전에 한 번만 읽어야 함)
    display_info = pygame.display.Info()
    real_screen_w = display_info.current_w
    real_screen_h = display_info.current_h

    # [전체화면] 시작은 창모드(900x600)로. F키 누르면 이벤트 루프에서 전체화면으로 전환됨
    is_fullscreen = False
    SCREEN_WIDTH  = WINDOWED_WIDTH
    SCREEN_HEIGHT = WINDOWED_HEIGHT
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("메모리움")
    clock = pygame.time.Clock()

    # ── 미디어 로드 ──────────────────────────
    image_paths = load_images(PHOTOS_DIR)
    video_paths = load_videos(VIDEOS_DIR)
    if not image_paths:
        print("photos 폴더에 이미지가 없어요!")
        sys.exit()
    all_media = image_paths + video_paths
    random.shuffle(all_media)
    print(f"사진 {len(image_paths)}장 + 영상 {len(video_paths)}개 불러옴")

    # ── 음악 로드 ──────────────────────────
    music_paths = load_music(MUSIC_DIR)
    music_calm_paths = load_music(MUSIC_CALM_DIR)   # 썬다운 모드용 차분한 음악
    music_index = 0
    if not music_paths:
        print("음악 없음 — 배경음악 없이 실행")
    if not music_calm_paths:
        print("차분한 음악(music_calm) 없음 — 썬다운 시 음악 전환 생략됨")
    # 시작할 때는 곧바로 재생하지 않음 — 아래 메인 루프의 통합 판단부에서
    # "지금 음악이 켜져 있어야 하는 상황인지"를 보고 자동으로 시작함
    music_currently_playing = False   # 지금 실제로 재생 중인지 추적

    # ── 웹캠 초기화 ──────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠 없음 — 표정 인식 없이 실행")
        cap = None

    # [최적화] 표정 분석 결과를 주고받을 공유 딕셔너리 + 분석 전용 스레드 시작
    # (분석 자체는 이 스레드 안에서만 일어나고, 메인 루프는 결과값만 읽어감)
    face_analysis_state = {"confidence": 0, "happy": 0, "new": False, "lock": threading.Lock()}
    if cap:
        face_thread = threading.Thread(
            target=start_face_analysis_thread,
            args=(cap, face_analysis_state),
            daemon=True   # 메인 프로그램 종료 시 같이 종료
        )
        face_thread.start()

    # ── 기억 시스템 로드 ────────────────────
    memory = load_memory()   # 태그 가중치 불러오기
    print(f"기억 불러옴: {memory}")

    # ── 태그 시스템 로드 ────────────────────
    global _tags_cache
    _tags_cache = load_tags()  # tags.json 불러오기
    print(f"태그 불러옴: {len(_tags_cache)}개 파일")
    tags_last_modified     = os.path.getmtime(TAGS_FILE)     if os.path.exists(TAGS_FILE)     else 0
    settings_last_modified = os.path.getmtime(SETTINGS_FILE) if os.path.exists(SETTINGS_FILE) else 0
    slide_interval_ms = [SLIDE_INTERVAL]  # 리스트로 감싸서 루프 안에서 수정 가능하게
    # 썬다운 설정값도 리스트로 감싸서 실시간 반영 가능하게
    # [enabled, hour, minute, end_hour, end_minute, auto_music, face_auto_music]
    sundowner_ref = [
        SUNDOWNER_ENABLED, SUNDOWNER_HOUR, SUNDOWNER_MINUTE,
        SUNDOWNER_END_HOUR, SUNDOWNER_END_MINUTE,
        SUNDOWNER_AUTO_MUSIC, FACE_AUTO_MUSIC
    ]

    # ── 알림 상태 ──────────────────────────
    fired_medicines = set()   # 이미 울린 복약 알림 기록
    fired_chimes    = set()   # 이미 울린 시보 기록

    # ── 썬다운 모드 상태 ──────────────────────
    sundowner_active = False   # 현재 썬다운 모드 활성화 여부

    # ── 얼굴/표정 인식 상태 ──────────────────
    face_detected = False   # 현재 얼굴 감지 여부
    greeted       = False   # 이번 활성화에서 인사했는지 여부
    no_face_count = 0       # 연속으로 얼굴 못 찾은 횟수
    smile_count   = 0       # 현재 미디어에서 웃음 감지 횟수
    frame_count   = 0       # 프레임 카운터 (표정 분석 간격 조절)
    viewing_start_time = None   # 응시(얼굴 감지) 시작 시각 — 응시 기록용

    # STT 스레드와 face_detected를 공유하기 위한 리스트
    # (bool을 직접 넘기면 복사본이 전달되므로, 리스트로 감싸서 공유)
    face_detected_ref = [False]

    # STT 리스너 백그라운드 스레드 시작
    stt_thread = threading.Thread(
        target=start_stt_listener,
        args=(face_detected_ref,),
        daemon=True   # 메인 프로그램 종료 시 같이 종료
    )
    stt_thread.start()

    # ── 선호 태그 상태 ──────────────────────
    prefer_tag   = None   # 현재 선호 태그 (웃음 감지 시 설정)
    prefer_count = 0      # 선호 태그 미디어 재생 횟수 (PREFER_MAX 도달 시 초기화)

    # ── 테두리 밝기 상태 ──────────────────────
    border_alpha = BORDER_MAX   # 시작 시 테두리 어두움

    # ── 슬라이드쇼 상태 ──────────────────────
    cap_v         = None    # 현재 재생 중인 영상 캡처 객체
    current_index = random.randrange(len(all_media))
    last_switch   = pygame.time.get_ticks()
    next_img      = None    # 슬라이드 전환 중 다음 사진
    next_index    = None
    slide_offset  = 0       # 슬라이드 애니메이션 오프셋
    is_sliding    = False   # 슬라이드 전환 중 여부

    # 첫 미디어 로드
    if is_video(all_media[current_index]):
        cap_v = cv2.VideoCapture(all_media[current_index])
        current_img = None
    else:
        current_img = fit_image(pygame.image.load(all_media[current_index]), SCREEN_WIDTH, SCREEN_HEIGHT)

    # =============================================
    # 메인 루프
    # =============================================
    running = True
    while running:

        # ── [0] 설정/태그 파일 변경 감지 ──────────
        if os.path.exists(TAGS_FILE):
            mtime = os.path.getmtime(TAGS_FILE)
            if mtime != tags_last_modified:
                _tags_cache = load_tags()
                tags_last_modified = mtime
                print(f"태그 실시간 반영: {len(_tags_cache)}개 파일")

        if os.path.exists(SETTINGS_FILE):
            smtime = os.path.getmtime(SETTINGS_FILE)
            if smtime != settings_last_modified:
                s = load_settings()
                MEDICINE_TIMES[:]      = [(t["hour"], t["minute"]) for t in s.get("medicine_times", [])]
                CHIME_HOURS[:]         = list(range(7, 22)) if s.get("chime_enabled", True) else []
                slide_interval_ms[0]   = s.get("slide_interval", 10) * 1000
                # 썬다운 설정도 실시간 반영
                sundowner_ref[0]  = s.get("sundowner_enabled", True)
                sundowner_ref[1]  = s.get("sundowner_hour", 19)
                sundowner_ref[2]  = s.get("sundowner_minute", 0)
                sundowner_ref[3]  = s.get("sundowner_end_hour", 21)
                sundowner_ref[4]  = s.get("sundowner_end_minute", 0)
                sundowner_ref[5]  = s.get("sundowner_auto_music", True)
                sundowner_ref[6]  = s.get("face_auto_music", True)
                settings_last_modified = smtime
                print(f"설정 실시간 반영 — 복약: {MEDICINE_TIMES}, 썬다운: {sundowner_ref[1]}시{sundowner_ref[2]}분~{sundowner_ref[3]}시{sundowner_ref[4]}분")

        # ── [1] 이벤트 처리 ──────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:    # ESC: 종료
                    running = False
                if event.key == pygame.K_RIGHT and not is_sliding:  # →: 다음 미디어
                    next_index = pick_next(all_media, current_index, prefer_tag, force=True)  # 선호 태그 100% 보장
                    next_path  = all_media[next_index]
                    if is_video(next_path):
                        if cap_v: cap_v.release()
                        current_index = next_index
                        cap_v = cv2.VideoCapture(next_path)
                        current_img = None
                    else:
                        next_img = fit_image(pygame.image.load(next_path), SCREEN_WIDTH, SCREEN_HEIGHT)
                        slide_offset = 0
                        is_sliding   = True
                    last_switch = pygame.time.get_ticks()
                if event.key == pygame.K_m:   # M: 음악 켜기/끄기 (수동 의사만 기록, 실제 재생은 아래 통합 판단부가 처리)
                    music_on = not music_on
                    print("음악 수동 켜짐" if music_on else "음악 수동 꺼짐")
                if event.key == pygame.K_f:   # F: 전체화면 ↔ 창모드 전환
                    is_fullscreen = not is_fullscreen
                    if is_fullscreen:
                        # main() 맨 처음(창 만들기 전)에 저장해둔 진짜 모니터 해상도 사용
                        SCREEN_WIDTH  = real_screen_w
                        SCREEN_HEIGHT = real_screen_h
                        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
                        print(f"전체화면 전환 ({SCREEN_WIDTH}x{SCREEN_HEIGHT})")
                    else:
                        SCREEN_WIDTH  = WINDOWED_WIDTH
                        SCREEN_HEIGHT = WINDOWED_HEIGHT
                        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                        print("창모드 전환")
                    # 화면 크기가 바뀌었으니 현재 보고 있는 사진도 새 크기로 다시 맞춤
                    # (영상은 매 프레임마다 SCREEN_WIDTH/HEIGHT를 다시 읽어서 그리므로 별도 처리 불필요)
                    if not is_video(all_media[current_index]):
                        current_img = fit_image(pygame.image.load(all_media[current_index]), SCREEN_WIDTH, SCREEN_HEIGHT)

            if event.type == MUSIC_END_EVENT:  # 곡 끝 → 다음 곡 (이미 재생 중이었으므로 그대로 이어서 재생)
                # 썬다운 모드 중이면 차분한 음악 목록에서, 아니면 평소 음악 목록에서 다음 곡
                current_list = music_calm_paths if (sundowner_active and music_calm_paths) else music_paths
                if current_list:
                    music_index = (music_index + 1) % len(current_list)
                    play_music(current_list, music_index, volume=(0.3 if sundowner_active else 0.5))

        # ── [2] 표정 인식 (백그라운드 스레드가 분석해둔 최신 결과만 읽음) ──
        # DeepFace.analyze()는 더 이상 여기서 직접 안 부름 — 무거운 연산은
        # start_face_analysis_thread()가 별도 스레드에서 처리하고, 여기선
        # 그 결과값(confidence, happy_score)만 가져다 씀. 그래서 분석이
        # 아무리 오래 걸려도 이 아래 화면 그리기/음악 재생은 안 밀림.
        if cap:
            with face_analysis_state["lock"]:
                has_new     = face_analysis_state["new"]
                confidence  = face_analysis_state["confidence"]
                happy_score = face_analysis_state["happy"]
                face_analysis_state["new"] = False   # 읽었으니 새 결과 표시 끔

            if has_new:
                new_face = confidence > FACE_CONFIDENCE

                # 얼굴 감지/사라짐 처리 (음악 켜고 끄기는 더 이상 여기서 안 함 —
                # 아래 메인 루프의 통합 판단부가 face_detected 값을 보고 알아서 처리함)
                if new_face:
                    no_face_count = 0
                    if not face_detected:
                        face_detected = True
                        face_detected_ref[0] = True   # STT 스레드에 얼굴 감지 알림
                        viewing_start_time = datetime.datetime.now()
                        if not greeted:
                            speak(get_greeting_text(), priority=3)
                            greeted = True
                            print("얼굴 감지")
                else:
                    no_face_count += 1
                    if no_face_count >= NO_FACE_THRESHOLD and face_detected:
                        face_detected = False
                        face_detected_ref[0] = False   # STT 스레드에 얼굴 사라짐 알림
                        greeted       = False
                        no_face_count = 0
                        prefer_tag    = None
                        prefer_count  = 0
                        smile_count   = 0
                        if viewing_start_time is not None:
                            add_viewing_record(viewing_start_time, all_media[current_index])
                            viewing_start_time = None
                        print("얼굴 사라짐 — 테두리 어두워짐")

                # 웃음 감지 → 선호 태그 설정
                if happy_score >= HAPPY_THRESHOLD and face_detected:
                    smile_count += 1
                    print(f"웃음 ({happy_score:.1f}%) {smile_count}회 | {os.path.basename(all_media[current_index])}")
                    if smile_count >= SMILE_NEEDED and prefer_tag is None:
                        # 중복 태그 중 첫 번째를 선호 태그로 설정
                        prefer_tag   = get_tags(all_media[current_index])[0]
                        prefer_count = 0   # 설정 후 카운트 시작
                        # 모든 태그에 가중치 +1
                        for t in get_tags(all_media[current_index]):
                            memory = add_memory(memory, t)
                        print(f"선호 태그 설정: {prefer_tag} (태그: {get_tags(all_media[current_index])})")

        # ── [3] 테두리 밝기 서서히 전환 ──────────
        target_border = BORDER_MIN if face_detected else BORDER_MAX
        if border_alpha < target_border:
            border_alpha = min(border_alpha + ALPHA_STEP, target_border)
        elif border_alpha > target_border:
            border_alpha = max(border_alpha - ALPHA_STEP, target_border)

        # ── [4] 시간 체크 (시보 + 복약 알림) ────
        now = datetime.datetime.now()
        ch, cm = now.hour, now.minute

        if CHIME_ENABLED and cm == 0 and ch in CHIME_HOURS:   # 정각 시보
            if ch not in fired_chimes:
                speak(get_time_text(), priority=2)  # 시보 — 중간 우선순위
                fired_chimes.add(ch)
                print(f"시보: {get_time_text()}")
        else:
            fired_chimes.discard(ch)

        for h, m in MEDICINE_TIMES:         # 복약 알림
            if ch == h and cm == m:
                if (h, m) not in fired_medicines:
                    speak("약 드실 시간이에요. 잊지 마세요!", priority=1)  # 복약 — 최고 우선순위
                    fired_medicines.add((h, m))
                    print(f"복약 알림: {h}시 {m}분")
            else:
                fired_medicines.discard((h, m))

        # ── 썬다운 모드 체크 ──────────────────
        if sundowner_ref[0]:   # sundowner_enabled
            now_minutes       = ch * 60 + cm
            sundowner_start   = sundowner_ref[1] * 60 + sundowner_ref[2]   # hour, minute
            sundowner_end     = sundowner_ref[3] * 60 + sundowner_ref[4]   # end_hour, end_minute
            in_sundown_period = sundowner_start <= now_minutes < sundowner_end

            if in_sundown_period and not sundowner_active:
                sundowner_active = True
                print(f"썬다운 모드 시작 ({sundowner_ref[1]}시 {sundowner_ref[2]}분 ~ {sundowner_ref[3]}시 {sundowner_ref[4]}분)")
                if music_currently_playing and music_calm_paths:
                    pygame.mixer.music.fadeout(2000)
                    music_index = 0
                    play_music(music_calm_paths, music_index, fade_in_ms=3000, volume=0.3)
            elif not in_sundown_period and sundowner_active:
                sundowner_active = False
                print("썬다운 모드 종료")
                if music_currently_playing and music_paths:
                    pygame.mixer.music.fadeout(2000)
                    music_index = 0
                    play_music(music_paths, music_index, fade_in_ms=3000, volume=0.5)
        else:
            # 썬다운 비활성화 상태에서 혹시 켜져있으면 종료
            if sundowner_active:
                sundowner_active = False

        # ── 음악 재생 여부 통합 판단 ──────────
        if not music_on:
            should_play_music = False
        elif sundowner_active and sundowner_ref[5]:   # sundowner_auto_music
            should_play_music = True
        elif (not sundowner_active) and face_detected and sundowner_ref[6]:   # face_auto_music
            should_play_music = True
        else:
            should_play_music = False

        if should_play_music and not music_currently_playing:
            # 꺼져 있다가 켜야 하는 상황 — 현재 시간대에 맞는 곡 목록으로 재생 시작
            target_list = music_calm_paths if (sundowner_active and music_calm_paths) else music_paths
            if target_list:
                play_music(target_list, music_index, volume=(0.3 if sundowner_active else 0.5))
                music_currently_playing = True
        elif not should_play_music and music_currently_playing:
            # 켜져 있다가 꺼야 하는 상황
            pygame.mixer.music.pause()
            music_currently_playing = False

        # ── [5] 영상 재생 ──────────────────────
        if cap_v is not None:
            video_done = play_video_frame(cap_v, screen, border_alpha)
            if video_done:
                smile_count = 0
                frame_count = 0
                cap_v.release()
                cap_v = None

                if prefer_tag is not None:
                    if prefer_count < PREFER_MAX:
                        # 아직 추가 미디어 더 보여줘야 함 — 선호 태그 유지
                        prefer_count += 1
                        print(f"선호 태그 미디어 {prefer_count}/{PREFER_MAX}장")
                        current_index = pick_next(all_media, current_index, prefer_tag, force=True)  # 선호 태그 100% 보장
                    else:
                        # 2장 다 봤음 — 랜덤 전환
                        prefer_tag   = None
                        prefer_count = 0
                        print("선호 태그 완료 — 랜덤 전환")
                        current_index = pick_next(all_media, current_index, None, memory=memory, sundowner_active=sundowner_active)
                else:
                    current_index = pick_next(all_media, current_index, None)

                path = all_media[current_index]
                if is_video(path):
                    cap_v = cv2.VideoCapture(path)
                    current_img = None
                else:
                    current_img = fit_image(pygame.image.load(path), SCREEN_WIDTH, SCREEN_HEIGHT)
                last_switch = pygame.time.get_ticks()
                print(f"미디어 전환: {os.path.basename(path)}")
            clock.tick(30)
            continue   # 영상 재생 중엔 사진 그리기 스킵

        # ── [6] 자동 사진 넘기기 ──────────────
        ticks_now = pygame.time.get_ticks()
        if ticks_now - last_switch >= slide_interval_ms[0] and not is_sliding:
            smile_count = 0
            frame_count = 0

            if prefer_tag is not None:
                if prefer_count < PREFER_MAX:
                    # 아직 추가 미디어 더 보여줘야 함 — 선호 태그 유지
                    prefer_count += 1
                    print(f"선호 태그 미디어 {prefer_count}/{PREFER_MAX}장")
                    next_index = pick_next(all_media, current_index, prefer_tag, force=True)  # 선호 태그 100% 보장
                else:
                    # 2장 다 봤음 — 랜덤 전환
                    prefer_tag   = None
                    prefer_count = 0
                    print("선호 태그 완료 — 랜덤 전환")
                    next_index = pick_next(all_media, current_index, None, memory=memory, sundowner_active=sundowner_active)
            else:
                next_index = pick_next(all_media, current_index, None)

            next_path = all_media[next_index]
            tag = get_tag(next_path)

            if is_video(next_path):   # 다음이 영상이면 바로 재생
                current_index = next_index
                cap_v = cv2.VideoCapture(next_path)
                current_img = None
                last_switch = ticks_now
                print(f"영상 재생: {os.path.basename(next_path)} (태그: {tag})")
            else:                     # 다음이 사진이면 슬라이드 애니메이션
                next_img = fit_image(pygame.image.load(next_path), SCREEN_WIDTH, SCREEN_HEIGHT)
                slide_offset = 0
                is_sliding   = True
                last_switch  = ticks_now
                print(f"사진 전환: {os.path.basename(next_path)} (태그: {tag})")

        # ── [7] 슬라이드 애니메이션 진행 ─────────
        if is_sliding and next_img is not None:
            slide_offset += SLIDE_SPEED
            if slide_offset >= SCREEN_WIDTH:   # 전환 완료
                current_img   = next_img
                current_index = next_index
                next_img      = None
                slide_offset  = 0
                is_sliding    = False

        # ── [8] 화면 그리기 (사진) ─────────────
        if current_img is not None:
            screen.fill(BG_COLOR)
            img_w, img_h = current_img.get_size()
            cx = (SCREEN_WIDTH - img_w) // 2
            cy = (SCREEN_HEIGHT - img_h) // 2

            if is_sliding and next_img is not None:
                # 슬라이드 전환 중: 현재 사진 왼쪽으로, 다음 사진 오른쪽에서 들어옴
                screen.blit(current_img, (cx - slide_offset, cy))
                nw, nh = next_img.get_size()
                screen.blit(next_img, ((SCREEN_WIDTH - nw) // 2 + SCREEN_WIDTH - slide_offset, cy))
            else:
                screen.blit(current_img, (cx, cy))

            draw_border(screen, border_alpha)   # 테두리 오버레이
            draw_date(screen, get_date_text(), 255)   # 날짜는 항상 선명하게 고정 표시
            pygame.display.flip()

        clock.tick(30)   # 초당 30프레임

    # ── 종료 처리 ──────────────────────────
    if viewing_start_time is not None:
        add_viewing_record(viewing_start_time, all_media[current_index])  # 응시 중에 종료됐다면 마지막 기록도 저장
    if cap:   cap.release()
    if cap_v: cap_v.release()
    pygame.quit()


def run_flask():
    """Flask 웹 설정 서버를 백그라운드 스레드로 실행"""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "app", os.path.join(os.path.dirname(__file__), "app.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
    except Exception as e:
        print(f"웹 서버 실행 실패: {e}")


if __name__ == "__main__":
    # 프로그램 시작할 때 오래된 말씀 기록 한 번 정리 (90일 넘은 것들 보관 파일로 이동)
    archive_old_speech(threshold_days=90)

    # Flask 서버 백그라운드로 실행
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print("웹 설정 페이지: http://localhost:5000")
    main()
