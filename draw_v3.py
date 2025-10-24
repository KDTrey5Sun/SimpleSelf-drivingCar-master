import pygame
import numpy as np

# 画布和赛道参数
WIDTH, HEIGHT = 1516, 800
# WIDTH, HEIGHT = 800, 600
TRACK_WIDTH = 200
LANE_WIDTH = 20
CENTER_X = WIDTH // 2

def draw_track(win):
    BLACK = (0, 0, 0)
    GRAY = (100, 100, 100)
    GREEN = (0, 180, 0)
    RED = (255, 0, 0)
    YELLOW = (255, 255, 0)
    WHITE = (255, 255, 255)

    win.fill(BLACK)
    # 绿色内道
    pygame.draw.rect(win, GREEN, (CENTER_X - TRACK_WIDTH//2 + LANE_WIDTH, 0, TRACK_WIDTH - 2*LANE_WIDTH, HEIGHT))
    # 灰色赛道
    pygame.draw.rect(win, GRAY, (CENTER_X - TRACK_WIDTH//2, 0, TRACK_WIDTH, HEIGHT))
    # 左红黄护栏
    for i in range(0, HEIGHT, 40):
        color = RED if (i // 40) % 2 == 0 else YELLOW
        pygame.draw.rect(win, color, (CENTER_X - TRACK_WIDTH//2, i, LANE_WIDTH, 40))
    # 右红黄护栏
    for i in range(0, HEIGHT, 40):
        color = RED if (i // 40) % 2 == 0 else YELLOW
        pygame.draw.rect(win, color, (CENTER_X + TRACK_WIDTH//2 - LANE_WIDTH, i, LANE_WIDTH, 40))
    # 黑白格终点线
    finish_line_y = 40
    finish_line_height = 16
    square_size = 10
    start_x = CENTER_X - TRACK_WIDTH//2 + LANE_WIDTH
    end_x = CENTER_X + TRACK_WIDTH//2 - LANE_WIDTH
    for row in range(2):  # 两行
        for i, x in enumerate(range(start_x, end_x, square_size)):
            color = WHITE if (i + row) % 2 == 0 else BLACK
            pygame.draw.rect(win, color, (x, finish_line_y + row * square_size, square_size, square_size))

def draw(win, car):
    draw_track(win)
    car.draw(win)
    pygame.display.update()




pygame.init()
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Test Track Drawing")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    draw_track(win)
    pygame.display.update()

pygame.quit()


