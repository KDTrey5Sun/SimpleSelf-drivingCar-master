import math
import os
import pygame
import numpy as np
from draw_v3 import draw, draw_track, WIDTH, HEIGHT, TRACK_WIDTH, LANE_WIDTH, CENTER_X

CENTER_X = CENTER_X  # 从 draw_v3 导入
MID_TRACK = TRACK_WIDTH / 2

def scale_image(img, factor):
    w, h = img.get_width(), img.get_height()
    size = (max(1, int(w * factor)), max(1, int(h * factor)))
    return pygame.transform.scale(img, size)

# 资源加载（windowless 下也需要 Surface，对读取失败做降级）
def _load_image(path, color=(255, 0, 0), size=(40, 20)):
    try:
        img = pygame.image.load(path)
        return img.convert_alpha()
    except Exception:
        surf = pygame.Surface(size, pygame.SRCALPHA)
        surf.fill(color)
        return surf

RED_CAR = scale_image(_load_image("imgs/red-car.png", (200, 0, 0)), 0.4)
GREEN_CAR = scale_image(_load_image("imgs/green-car.png", (0, 200, 0)), 0.3)
CENTER_CAR = scale_image(_load_image("imgs/green-car.png", (0, 200, 0), (10, 5)), 0.05)

FPS = 30

class AbstractCar:
    def __init__(self, max_vel, rotation_vel):
        self.img = GREEN_CAR
        self.max_vel = float(max_vel)
        self.rotation_vel = float(rotation_vel)     # 每步旋转角度（度）
        self.vel = 0.0
        self.acc = 0.6
        self.brake = 0.8
        self.angle = 0.0        # 0 度表示朝上
        self.x, self.y = CENTER_X, HEIGHT - 80
        self.START_POS = (CENTER_X, HEIGHT - 80)
        self.prev_x, self.prev_y = self.x, self.y

    def rotate(self, left=False, right=False):
        if left:
            self.angle += self.rotation_vel
        elif right:
            self.angle -= self.rotation_vel
        # 归一化角度
        if self.angle > 180: self.angle -= 360
        if self.angle <= -180: self.angle += 360

    def draw(self, win):
        # 训练脚本 windowless 不用画；此函数供可视化时使用
        rotated_image = pygame.transform.rotate(self.img, self.angle)
        new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
        win.blit(rotated_image, new_rect.topleft)

    def move_forward(self):
        self.vel = min(self.vel + self.acc, self.max_vel)

    def move_backward(self):
        self.vel = max(self.vel - self.brake, 0.0)

    def move(self):
        # 根据朝向移动（角度以度为单位；0 度向上）
        rad = math.radians(self.angle)
        dx = -math.sin(rad) * self.vel
        dy = -math.cos(rad) * self.vel
        self.prev_x, self.prev_y = self.x, self.y
        self.x += dx
        self.y += dy

    def collide(self):
        # 轨道内侧边界
        left = CENTER_X - TRACK_WIDTH / 2 + LANE_WIDTH
        right = CENTER_X + TRACK_WIDTH / 2 - LANE_WIDTH
        if self.x < left or self.x > right:
            return True
        if self.y < 0 or self.y > HEIGHT:
            return True
        return False

    def reset(self):
        self.x, self.y = self.START_POS
        self.vel = 0.0
        self.angle = 0.0
        self.prev_x, self.prev_y = self.x, self.y

def blit_rotate_center(win, image, top_left, angle):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center=image.get_rect(topleft=top_left).center)
    win.blit(rotated_image, new_rect.topleft)

class ComputerCar(AbstractCar):
    IMG = GREEN_CAR
    START_POS = (CENTER_X, HEIGHT - 80)

    def __init__(self, max_vel, rotation_vel):
        super().__init__(max_vel, rotation_vel)
        self.img = self.IMG
        self.cumulated_rewards = 0.0
        self.is_finished = False
        self.is_collide = False
        self.START_POS = (CENTER_X, HEIGHT - 80)

    def _normalize(self, v, denom):
        return float(np.clip(v / max(1e-6, denom), -1.0, 1.0))

    def get_state(self):
        # 归一化状态: [横向偏移, 纵向进度, 速度, 角度]
        lateral = (self.x - CENTER_X) / (TRACK_WIDTH / 2.0)     # [-1,1] 左右偏移
        progress = 1.0 - (self.y / HEIGHT)                      # 0(起点) -> 1(终点)
        speed = self.vel / max(1.0, self.max_vel)
        angle_norm = self.angle / 90.0                          # 粗略规范到 [-2,2]
        return np.array([lateral, progress, speed, angle_norm], dtype=np.float32)

    def get_distance_to_border(self):
        left = CENTER_X - TRACK_WIDTH / 2 + LANE_WIDTH
        right = CENTER_X + TRACK_WIDTH / 2 - LANE_WIDTH
        dist_left = self.x - left
        dist_right = right - self.x
        return dist_left, dist_right

    def get_distance_delta(self):
        # 与中心线的偏移（正右负左）
        return self.x - CENTER_X

    def reset(self):
        super().reset()
        self.cumulated_rewards = 0.0
        self.is_finished = False
        self.is_collide = False

    def reset_env(self):
        self.reset()
        return self.get_state()

    def step(self, action):
        # 动作：0 直行、1 左转前进、2 右转前进、3 减速（或短暂后退）
        if action == 1:
            self.rotate(left=True)
            self.move_forward()
        elif action == 2:
            self.rotate(right=True)
            self.move_forward()
        elif action == 3:
            self.move_backward()
        else:
            self.move_forward()

        self.move()

        # 撞墙终止
        if self.collide():
            self.is_collide = True
            reward = -500.0
            self.cumulated_rewards += reward
            return self.get_state(), reward, True

        # 终点检测：位于顶部一道终点带内
        finish_y = 40
        finish_h = 25
        left = CENTER_X - TRACK_WIDTH / 2 + LANE_WIDTH
        right = CENTER_X + TRACK_WIDTH / 2 - LANE_WIDTH
        car_rect = self.img.get_rect(center=(self.x, self.y))
        finish_rect = pygame.Rect(left, finish_y, right - left, finish_h)
        if car_rect.colliderect(finish_rect):
            self.is_finished = True
            reward = 3000.0
            print("FINISH!")
            print("reward:", reward)
            self.cumulated_rewards += reward
            return self.get_state(), reward, True

        # 中间奖励：向上前进 + 保持在中心
        dy = self.prev_y - self.y                         # 向上移动为正



        progress_r = 10.0 * max(0.0, dy)                  # 进度奖励
        lateral_penalty = -2.0 * abs(self.x - CENTER_X) / (TRACK_WIDTH / 2.0)
        steering_penalty = -0.01 * abs(self.angle)        # 大角度惩罚（轻微）
        reward = progress_r + lateral_penalty + steering_penalty





        self.cumulated_rewards += reward
        return self.get_state(), reward, False