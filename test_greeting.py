import sys
import os

# slideshow_tts.py와 같은 폴더에서 실행해야 함
sys.path.insert(0, os.path.dirname(__file__))

# slideshow_tts.py에서 함수만 가져오기 (실제 파일은 전혀 안 바뀜)
from slideshow_tts import get_greeting_text

# =============================================
# 시간대별 인사 멘트 테스트
# get_greeting_text(시각)을 직접 호출해서
# 컴퓨터 시계를 건드리지 않고도 결과를 바로 확인함
# =============================================

test_hours = [7, 9, 11, 13, 15, 17, 18, 20, 23]

print("=" * 40)
print("시간대별 인사 멘트 테스트")
print("=" * 40)

for hour in test_hours:
    greeting = get_greeting_text(hour)
    print(f"{hour:2d}시 -> {greeting}")

print("=" * 40)
print("테스트 끝! 위 결과가 의도한 멘트와 맞는지 확인해줘.")
