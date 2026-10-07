# Memorium — 기억을 돕는 AI 스마트 액자

🔗 **Portfolio**: https://subinlu22.github.io

치매 어르신의 기억을 돕는 **라즈베리파이 기반 AI 스마트 액자**입니다.
가족 사진·영상 슬라이드쇼, 음성 안내, 복약 알림, 표정 분석을 통해 어르신이 편안하게 추억을 떠올리도록 돕습니다.
제24회 임베디드 소프트웨어 경진대회 자유공모 부문 출품 (2026.08, 1인)

> 📷 데모 영상 / 스크린샷 자리

## 설계 원칙: "틀림을 드러내지 않는 UI"
- 오답·반복·혼란 상황에서도 어르신의 자존감을 지키는 방식으로만 반응
- 빨간색, X 표시, 경고음 없음. 밝게 정답만 자연스럽게 안내
- 같은 말을 반복하셔도 "맞아요, 그러셨군요" 공감 먼저 ("아까 하셨잖아요"는 절대 사용 안 함)
- 퀴즈 형식 대신 표정 분석으로 기억 인지 반응 확인

## 주요 기능
- 사진·영상 슬라이드쇼 + 음성 안내 (TTS)
- 복약 알림, 시보
- 표정 분석 (DeepFace)
- 웹 설정 페이지 (가족이 사진·설정 관리)

## 기술 스택
| 구분 | 사용 |
|---|---|
| 하드웨어 | Raspberry Pi 5, M.2 HAT+, Hailo AI 모듈, 모니터 |
| 언어 / 서버 | Python, Flask |
| AI | DeepFace, Gemini API |
| 음성 | gTTS, SpeechRecognition, PyAudio |
| 기타 | OpenCV, pygame |

## 라즈베리파이 이전 과정에서 해결한 것
| 문제 | 해결 |
|---|---|
| Hailo 모듈이 인식 안 됨 (`lspci`에 안 뜸) | M.2 HAT+는 PCIe 수동 활성화 필요 → `config.txt`에 `dtparam=pciex1` 추가 |
| 학교 네트워크에서 SSH/SCP 불가 | 유선·무선 대역이 분리되어 있음 → 모니터 직결로 작업, 데이터는 클라우드 경유 이전 |
| PyAudio 설치 실패 | `portaudio19-dev` 먼저 설치 |
| 브라우저 한글이 □로 깨짐 | 한글 폰트(`fonts-unfonts-core`) 설치 |

## 실행
```bash
source ~/memorium_env/bin/activate
python slideshow_tts.py
# 웹 설정 페이지: http://localhost:5000
```
- `.env`에 `GEMINI_API_KEY` 필요
- 사진·영상·음악 폴더와 사용자 데이터 json은 개인정보라 저장소에 포함하지 않음

---
🔗 **Portfolio**: https://subinlu22.github.io
