import numpy as np
import pygame
import time
import math
from utils import scale_image, blit_rotate_center
import cv2

# 加载图像资源
GRASS = scale_image(pygame.image.load("imgs/grass.jpg"), 2.5)
TRACK = scale_image(pygame.image.load("imgs/track1.png"), 0.3)
TRACK_BORDER = scale_image(pygame.image.load("imgs/track_border1.png"), 0.3)
TRACK_BORDER_MASK = pygame.mask.from_surface(TRACK_BORDER)
FINISH = pygame.image.load("imgs/finish.png")
FINISH_MASK = pygame.mask.from_surface(FINISH)
FINISH_POSITION = (453, 410)
RED_CAR = scale_image(pygame.image.load("imgs/red-car.png"), 0.4)
GREEN_CAR = scale_image(pygame.image.load("imgs/green-car.png"), 0.3)
CENTER_CAR = scale_image(pygame.image.load("imgs/green-car.png"), 0.05)

# 设置窗口
WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Racing Game!")
FPS = 30


# 加载赛道中心线图像并提取白色像素点坐标
CENTER_LINE_IMG = cv2.imread("imgs/track1.png", cv2.IMREAD_GRAYSCALE)
CENTER_LINE_POINTS = np.column_stack(np.where(CENTER_LINE_IMG > 0))  # shape: [N, 2] -> [y, x]

# 加载原始赛道图片（未缩放）
track_img_raw = cv2.imread("imgs/track1.png", cv2.IMREAD_GRAYSCALE)

# 取图片中间一行，统计灰色赛道的宽度（假设灰色为128左右，或用实际灰度值区间）
middle_row = track_img_raw[track_img_raw.shape[0] // 2]
# 你可以根据实际灰度值调整阈值
track_pixels = np.where((middle_row > 100) & (middle_row < 200))[0]
if len(track_pixels) > 0:
    raw_track_width = track_pixels[-1] - track_pixels[0]
else:
    raw_track_width = 0

# 计算缩放后的赛道宽度
TRACK_SCALE = 0.3
TRACK_WIDTH = raw_track_width * TRACK_SCALE
print("缩放后赛道宽度（像素）:", TRACK_WIDTH)
# 缩放后赛道宽度（像素）: 437.7