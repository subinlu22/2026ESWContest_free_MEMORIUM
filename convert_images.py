from PIL import Image
import os

# =============================================
# photos 폴더의 모든 이미지를 진짜 jpg로 변환
# =============================================

PHOTOS_DIR = r"C:\Users\dev\memorium\photos"

for filename in os.listdir(PHOTOS_DIR):
    filepath = os.path.join(PHOTOS_DIR, filename)
    name, ext = os.path.splitext(filename)

    try:
        img = Image.open(filepath)
        img = img.convert("RGB")  # RGB로 변환 (투명도 제거)
        new_path = os.path.join(PHOTOS_DIR, name + ".jpg")
        img.save(new_path, "JPEG")

        # 원본이 jpg가 아니었으면 원본 삭제
        if ext.lower() != ".jpg":
            os.remove(filepath)
            print(f"변환 완료: {filename} → {name}.jpg")
        else:
            print(f"덮어쓰기 완료: {filename}")

    except Exception as e:
        print(f"실패: {filename} — {e}")

print("완료!")
