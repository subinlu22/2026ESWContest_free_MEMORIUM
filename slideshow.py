import pygame
import os
import sys

# =============================================
# 메모리움 — 슬라이드쇼 기본 테스트
# 사진 폴더의 이미지를 순서대로 화면에 표시
# =============================================

# 사진 폴더 경로
PHOTOS_DIR = r"C:\Users\dev\memorium\photos"

# 사진 넘기는 간격 (밀리초) — 5000 = 5초
SLIDE_INTERVAL = 5000

# 화면 크기
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 600

# 배경 색상 (검정)
BG_COLOR = (0, 0, 0)


def load_images(folder):
    """폴더에서 jpg/png 이미지 파일 목록 불러오기"""
    extensions = (".jpg", ".jpeg", ".png")
    images = []
    for f in os.listdir(folder):
        if f.lower().endswith(extensions):
            images.append(os.path.join(folder, f))
    return sorted(images)  # 파일명 순서로 정렬


def fit_image(surface, screen_w, screen_h):
    """이미지를 화면 크기에 맞게 비율 유지하면서 리사이즈"""
    img_w, img_h = surface.get_size()
    ratio = min(screen_w / img_w, screen_h / img_h)
    new_w = int(img_w * ratio)
    new_h = int(img_h * ratio)
    return pygame.transform.smoothscale(surface, (new_w, new_h))


def main():
    pygame.init()

    # 화면 생성
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("메모리움 슬라이드쇼 테스트")

    clock = pygame.time.Clock()

    # 이미지 목록 불러오기
    image_paths = load_images(PHOTOS_DIR)
    if not image_paths:
        print("photos 폴더에 이미지가 없어요!")
        sys.exit()

    print(f"사진 {len(image_paths)}장 불러옴")

    current_index = 0       # 현재 보여주는 사진 번호
    last_switch = pygame.time.get_ticks()  # 마지막으로 사진 넘긴 시각

    # 첫 번째 이미지 로드
    current_img = fit_image(pygame.image.load(image_paths[current_index]), SCREEN_WIDTH, SCREEN_HEIGHT)

    running = True
    while running:

        # [이벤트 처리] 창 닫기 또는 ESC 키 누르면 종료
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                # 오른쪽 방향키: 다음 사진
                if event.key == pygame.K_RIGHT:
                    current_index = (current_index + 1) % len(image_paths)
                    current_img = fit_image(pygame.image.load(image_paths[current_index]), SCREEN_WIDTH, SCREEN_HEIGHT)
                    last_switch = pygame.time.get_ticks()
                # 왼쪽 방향키: 이전 사진
                if event.key == pygame.K_LEFT:
                    current_index = (current_index - 1) % len(image_paths)
                    current_img = fit_image(pygame.image.load(image_paths[current_index]), SCREEN_WIDTH, SCREEN_HEIGHT)
                    last_switch = pygame.time.get_ticks()

        # [자동 넘기기] SLIDE_INTERVAL 지나면 다음 사진으로
        now = pygame.time.get_ticks()
        if now - last_switch >= SLIDE_INTERVAL:
            current_index = (current_index + 1) % len(image_paths)
            current_img = fit_image(pygame.image.load(image_paths[current_index]), SCREEN_WIDTH, SCREEN_HEIGHT)
            last_switch = now
            print(f"사진 전환 → {os.path.basename(image_paths[current_index])}")

        # [화면 그리기] 배경 검정으로 채우고 이미지 중앙에 표시
        screen.fill(BG_COLOR)
        img_w, img_h = current_img.get_size()
        x = (SCREEN_WIDTH - img_w) // 2   # 가로 중앙 정렬
        y = (SCREEN_HEIGHT - img_h) // 2  # 세로 중앙 정렬
        screen.blit(current_img, (x, y))

        pygame.display.flip()
        clock.tick(30)  # 초당 30프레임

    pygame.quit()


if __name__ == "__main__":
    main()
