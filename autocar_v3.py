import numpy as np
import pygame
import time
import math
from draw_v3 import draw, draw_track, WIDTH, HEIGHT, TRACK_WIDTH, LANE_WIDTH, CENTER_X

# 赛道参数（与draw.py、DQN_CAR_v3.py一致）
# WIDTH, HEIGHT = 800, 600
# TRACK_WIDTH = 200
# LANE_WIDTH = 20
CENTER_X = WIDTH // 2
MID_TRACK = TRACK_WIDTH / 2

# 小车图片资源
def scale_image(img, factor):
    size = round(img.get_width() * factor), round(img.get_height() * factor)
    return pygame.transform.scale(img, size)

RED_CAR = scale_image(pygame.image.load("imgs/red-car.png"), 0.4)
GREEN_CAR = scale_image(pygame.image.load("imgs/green-car.png"), 0.3)
CENTER_CAR = scale_image(pygame.image.load("imgs/green-car.png"), 0.05)

FPS = 30

class AbstractCar:
    def __init__(self, max_vel, rotation_vel):
        self.START_POS = (CENTER_X, HEIGHT - 80)
        self.img = self.IMG
        self.max_vel = max_vel
        self.vel = 1
        self.rotation_vel = rotation_vel
        self.angle = 0
        self.x, self.y = self.START_POS
        self.prev_x, self.prev_y = self.x, self.y  # 初始化前一位置
        self.acceleration = 0.2

    def rotate(self, left=False, right=False):
        if left:
            self.angle += self.rotation_vel
        elif right:
            self.angle -= self.rotation_vel
        if self.angle > 180:
            self.angle -= 360
        elif self.angle < -180:
            self.angle += 360

    def draw(self, win):
        blit_rotate_center(win, self.img, (self.x, self.y), self.angle)

    def move_forward(self):
        self.vel = min(self.vel + self.acceleration, self.max_vel)
        self.move()

    def move_backward(self):
        self.vel = max(self.vel - self.acceleration, -self.max_vel / 2)
        self.move()

    def move(self):
        radians = math.radians(self.angle)
        vertical = math.cos(radians) * self.vel
        horizontal = math.sin(radians) * self.vel
        self.y -= vertical
        self.x -= horizontal

    def collide(self):
        # 判断是否碰撞赛道边界
        left_border = CENTER_X - TRACK_WIDTH // 2 + LANE_WIDTH
        right_border = CENTER_X + TRACK_WIDTH // 2 - LANE_WIDTH
        car_rect = self.img.get_rect(center=(self.x, self.y))
        if car_rect.left < left_border or car_rect.right > right_border:
            return True
        if car_rect.top < 0 or car_rect.bottom > HEIGHT:
            return True
        return False

    def reset(self):
        self.x, self.y = self.START_POS
        self.angle = 0
        self.vel = 0
        self.prev_x, self.prev_y = self.x, self.y  # 重置时也重置 prev

def blit_rotate_center(win, image, top_left, angle):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center=image.get_rect(topleft=top_left).center)
    win.blit(rotated_image, new_rect.topleft)

class ComputerCar(AbstractCar):
    IMG = GREEN_CAR
    START_POS = (CENTER_X, HEIGHT - 80)

    def __init__(self, max_vel, rotation_vel):
        super().__init__(max_vel, rotation_vel)
        self.cumulated_rewards = 0
        self.is_finished = False
        self.is_collide = False

    def get_state(self):
        # 目标点为终点线中央
        finish_y = 40
        finish_x = CENTER_X

        # 计算小车与终点线的距离和角度差
        dx = finish_x - self.x
        dy = finish_y - self.y
        distance = math.sqrt(dx ** 2 + dy ** 2)

        angle_diff = math.atan2(dy, dx) * 180 / math.pi - self.angle
        angle_diff = (angle_diff + 180) % 360 - 180  # 限制在[-180,180]


        # 计算小车中心到左右护栏的距离
        border_dist = self.get_distance_to_border()
        center_dist = (MID_TRACK - border_dist) / MID_TRACK  # 标准化到[-1, 1]

        return np.array([
            angle_diff / 180,
            self.vel / self.max_vel,
            distance,
            center_dist
        ], dtype=np.float32)

    def get_distance_to_border(self):
        """
        计算小车中心到左右护栏的最短距离（像素单位）
        """
        left_border = CENTER_X - TRACK_WIDTH // 2 + LANE_WIDTH
        right_border = CENTER_X + TRACK_WIDTH // 2 - LANE_WIDTH
        car_rect = self.img.get_rect(center=(self.x, self.y))
        dist_left = self.x - left_border
        dist_right = right_border - self.x
        return min(dist_left, dist_right)

    def get_distance_delta(self):
        dx = self.x - self.prev_x
        dy = self.y - self.prev_y
        return math.sqrt(dx * dx + dy * dy)

    def reduce_speed(self):
        self.vel = max(self.vel - self.acceleration / 2, 0)
        self.move()

    def reset_env_v1(self):
        self.reset()
        return self.get_state()
    
    def reset_env(self):
        self.reset()
        return self.get_state()
    
   # ...existing code...

    def step(self, action):
        self.prev_x, self.prev_y = self.x, self.y

        if action == 0:
            self.rotate(left=True)
            self.move_forward()
        elif action == 1:
            self.rotate(right=True)
            self.move_forward()
        elif action == 2:
            self.move_forward()
        elif action == 3:
            self.reduce_speed()

        done = False
        reward = 0

        if self.collide():
            reward = -500  # 撞墙惩罚
            done = True
            return self.get_state(), reward, done

        # 终点检测（与draw.py黑白格终点线一致）
        finish_line_y = 40
        finish_line_height = 20  # 两行格子高度
        start_x = CENTER_X - TRACK_WIDTH//2 + LANE_WIDTH
        end_x = CENTER_X + TRACK_WIDTH//2 - LANE_WIDTH
        car_rect = self.img.get_rect(center=(self.x, self.y))




        finish_rect = pygame.Rect(start_x, finish_line_y, end_x - start_x, finish_line_height)
        if car_rect.colliderect(finish_rect):
            reward = 3000  # 终点奖励
            done = True
            print("FINISH!")
            print("reward:", reward)
            return self.get_state(), reward, done





        distance_delta = self.get_distance_delta()
        border_dist = self.get_distance_to_border()
        center_dist = MID_TRACK - border_dist

        center_reward = 100 * (1 - min(abs(center_dist) / MID_TRACK, 1.0))  # 靠近中心线奖励

        reward += 2  # 存活奖励
        reward += 2.0 * self.vel  # 速度奖励
        reward += center_reward

        if distance_delta > 0.1:
            reward += 3 * distance_delta  # 鼓励移动
        else:
            reward -= 200  # 原地不动惩罚
        
        return self.get_state(), reward, done

        