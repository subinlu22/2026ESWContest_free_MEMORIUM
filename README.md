# Memorium — 기억을 돕는 AI 스마트 액자

치매 어르신의 기억을 돕는 **라즈베리파이 기반 AI 스마트 액자**입니다. 가족 사진·영상 슬라이드쇼, 음성 안내, 복약 알림, 표정 분석을 제공합니다. 제24회 임베디드 소프트웨어 경진대회 자유공모 부문에 1인으로 출품했습니다.

🔗 [Portfolio](https://subinlu22.github.io) · [최종 보고서 (PDF)](https://subinlu22.github.io/assets/projects/memorium-final-report.pdf)

---

## 데모

| [![프레임 저하 현상](https://img.youtube.com/vi/ubAPk-1Mrrw/hqdefault.jpg)](https://youtube.com/shorts/ubAPk-1Mrrw) | [![개선 후 동작](https://img.youtube.com/vi/Lr6p-6LIiUQ/hqdefault.jpg)](https://youtube.com/shorts/Lr6p-6LIiUQ) |
|---|---|
| 라즈베리파이 환경에서의 실시간 프레임 저하 현상 | 프레임 저하 개선 후 동작 안정화 결과 |

## 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 제출 | 2026.08.19 최종 제출 |
| 인원 / 역할 | 1인 — 기획, 설계, 개발 전 과정 |
| 대상 | 치매 어르신과 가족 (가족이 사진·설정을 관리하고, 어르신은 액자로 추억을 봄) |
| 출품 | 제24회 임베디드 소프트웨어 경진대회 자유공모 부문 |

## 설계 원칙: "틀림을 드러내지 않는 UI"

오답·반복·혼란 상황에서도 어르신의 자존감을 지키는 방식으로만 반응합니다.

| 상황 | 반응 |
|---|---|
| 오답 | 빨간색·X·경고음을 사용하지 않고, 밝게 정답만 자연스럽게 안내 |
| 같은 말을 반복 | "맞아요, 그러셨군요" 공감을 먼저 하고 자연스럽게 전환 ("아까 하셨잖아요"는 사용하지 않음) |
| 퀴즈 형식 | 삭제하고, 표정 분석(DeepFace)으로 기억 인지 반응을 확인 |

## 설계 과정에서 바꾼 것

| 처음 방식 | 문제 | 바꾼 방식 |
|---|---|---|
| 퀴즈로 기억 확인 | 오답이 드러나는 구조라 어르신의 자존감에 부담 | 퀴즈를 삭제하고 표정 분석(DeepFace)으로 기억 인지 반응을 확인 |
| 반복 질문에 "아까 말씀하셨어요" 류 안내 | 반복 자체를 지적하게 됨 | 공감("맞아요, 그러셨군요")을 먼저 하고 자연스럽게 화제 전환 |

## 구현 현황

| 기능 | 상태 |
|---|---|
| 사진·영상 슬라이드쇼 | ✅ 사진 20장 + 영상 8개 불러오기 확인 |
| 음성 안내 (TTS) | ✅ |
| 복약 알림, 시보 | ✅ |
| 표정 분석 (DeepFace) | ✅ |
| 웹 설정 페이지 | ✅ `http://localhost:5000` |
| 대시보드 실데이터 연동, 날씨, 프로필, AI 요약 | ✅ |
| 라즈베리파이 5 이전 및 실기 구동 | ✅ (2026.07.30) |
| Hailo AI 모듈 인식 | ✅ `hailortcli fw-control identify` 정상 출력 |

## 하드웨어 · 환경

| 구분 | 내용 |
|---|---|
| 본체 | Raspberry Pi 5, Raspberry Pi OS (64-bit) |
| AI 가속 | M.2 HAT+ + Hailo AI 모듈 |
| 출력 | 모니터 |
| 언어 / 서버 | Python, Flask |
| AI / 음성 | DeepFace (표정 분석), Gemini API, gTTS, SpeechRecognition, PyAudio |
| 기타 | OpenCV, TensorFlow, pygame |
| 개발 | Windows에서 개발 후 라즈베리파이로 이전 |

## 라즈베리파이 설치

**1. PCIe 활성화 (M.2 HAT+는 수동 활성화가 필요합니다)**

`/boot/firmware/config.txt` 맨 아래에 추가 후 재부팅합니다.

```
dtparam=pciex1
```

```bash
hailortcli fw-control identify   # Firmware Version 등이 출력되면 정상 인식
```

**2. 시스템 패키지**

```bash
sudo apt install portaudio19-dev -y        # PyAudio 설치 전 필수
sudo apt install fonts-unfonts-core -y     # 한글 폰트 (없으면 브라우저 한글이 □로 표시)
```

**3. 가상환경과 라이브러리**

```bash
python3 -m venv memorium_env
source ~/memorium_env/bin/activate

pip install flask flask-cors opencv-python pillow numpy pandas requests python-dotenv
pip install pyaudio SpeechRecognition
pip install tensorflow deepface mtcnn retina-face
pip install pygame gTTS tf-keras           # 실행 중 추가로 필요했던 것들
```

**4. 코드와 데이터**

```bash
git clone https://github.com/subinlu22/memorium.git   # private 저장소는 Personal Access Token 필요
cd memorium
nano .env                                              # GEMINI_API_KEY 입력
```

`.gitignore`로 제외된 `photos/`, `videos/`, `music/`, `music_calm/` 폴더와 `memory.json`, `tags.json`, `settings.json`은 clone으로 내려오지 않으므로 별도로 옮깁니다.

## 실행

```bash
source ~/memorium_env/bin/activate
python slideshow_tts.py
# 웹 설정 페이지: http://localhost:5000
```

정상 실행 시 콘솔에 설정(복약 시각, 시보, 슬라이드 간격)과 불러온 사진·영상 수가 출력됩니다.

## 트러블슈팅

| 문제 | 원인 | 해결 |
|---|---|---|
| `lspci`에 Hailo 모듈이 보이지 않음 | M.2 HAT+는 AI HAT+ 일체형과 달리 PCIe를 수동으로 켜야 함 | `/boot/firmware/config.txt`에 `dtparam=pciex1` 추가 후 `hailortcli fw-control identify`로 인식 확인 |
| PC에서 라즈베리파이로 SSH·SCP 접속 불가 | 학교 PC는 유선 전용이고 학교 네트워크는 유선·무선 대역(VLAN)이 분리되어 서로 통신 불가 | 모니터·키보드 직결로 작업, 코드는 GitHub, git 제외 데이터(사진·영상·json)는 클라우드 드라이브를 거쳐 이전 |

## 프로젝트 구조

```
memorium/
├── slideshow_tts.py      # 메인 실행 파일 (슬라이드쇼, TTS, 복약 알림, 웹 서버)
├── .env                  # GEMINI_API_KEY (git 제외)
├── photos/ videos/       # 가족 사진·영상 (개인정보, git 제외)
├── music/ music_calm/    # 배경 음악 (git 제외)
└── memory.json  tags.json  settings.json   # 사용자 데이터 (git 제외)
```

---

🔗 [Portfolio](https://subinlu22.github.io)
