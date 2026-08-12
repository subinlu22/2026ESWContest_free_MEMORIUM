import cv2
from deepface import DeepFace

# =============================================
# 메모리움 — 표정 인식 단독 테스트
# 웹캠으로 얼굴 감지 + 표정 분류 → CMD에 출력
# ESC 키 누르면 종료
# =============================================

# 표정을 3가지로 단순화하는 함수
def classify_emotion(raw_emotion):
    """
    DeepFace가 반환하는 세부 표정을 3가지로 단순 분류
    happy → 밝음
    sad / angry / fear / disgust → 우울/불안
    neutral / surprise → 무반응
    """
    if raw_emotion in ["happy"]:
        return "😊 밝음"
    elif raw_emotion in ["sad", "angry", "fear", "disgust"]:
        return "😢 우울/불안"
    else:
        return "😐 무반응"


def main():
    # 웹캠 열기 (0 = 기본 웹캠)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("웹캠을 열 수 없어요!")
        return

    print("웹캠 시작 — ESC 키 누르면 종료")
    print("표정 분석 중...")

    # 매 프레임마다 분석하면 너무 느리니까 N프레임마다 한 번 분석
    frame_count = 0
    ANALYZE_EVERY = 15  # 15프레임마다 한 번 분석 (약 0.5초 간격)

    last_emotion = "분석 중..."  # 마지막으로 감지한 표정

    while True:
        ret, frame = cap.read()  # 웹캠에서 프레임 읽기
        if not ret:
            print("프레임을 읽을 수 없어요!")
            break

        frame_count += 1

        # ANALYZE_EVERY 프레임마다 표정 분석
        if frame_count % ANALYZE_EVERY == 0:
            try:
                # DeepFace로 표정 분석 (enforce_detection=False: 얼굴 못 찾아도 에러 안 냄)
                result = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    enforce_detection=False,
                    silent=True
                )
                raw_emotion = result[0]["dominant_emotion"]  # 가장 높은 확률의 표정
                emotion = classify_emotion(raw_emotion)
                last_emotion = emotion
                print(f"표정: {emotion} ({raw_emotion})")

            except Exception as e:
                last_emotion = "얼굴 미감지"

        # 화면에 현재 표정 텍스트 표시
        cv2.putText(
            frame,
            last_emotion,
            (30, 50),                    # 텍스트 위치
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,                         # 글자 크기
            (0, 255, 0),                 # 초록색
            2                            # 두께
        )

        cv2.imshow("메모리움 표정 인식 테스트", frame)  # 웹캠 화면 표시

        # ESC 키 누르면 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("종료")


if __name__ == "__main__":
    main()
