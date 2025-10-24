import numpy as np
import pygame
import time
import math
from utils import scale_image, blit_rotate_center
# import cv2

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
print(WIDTH, HEIGHT)

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Racing Game!")
FPS = 30

TRACK_WIDTH = 70
MID_TRACK = TRACK_WIDTH / 2

def get_track_border_points(surface):
    arr = pygame.surfarray.array3d(surface)
    arr = np.transpose(arr, (1, 0, 2))
    mask = np.any(arr != 0, axis=2)
    points = np.argwhere(mask)
    return points
TRACK_BORDER_POINTS = get_track_border_points(TRACK_BORDER)

class AbstractCar:
    def __init__(self, max_vel, rotation_vel):
        self.START_POS = (488, 370)
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

    def collide(self, mask, x=0, y=0):
        car_mask = pygame.mask.from_surface(self.img)
        offset = (int(self.x - x), int(self.y - y))
        poi = mask.overlap(car_mask, offset)
        return poi

    def reset(self):
        self.x, self.y = self.START_POS
        self.angle = 0
        self.vel = 0
        self.prev_x, self.prev_y = self.x, self.y  # 重置时也重置 prev


class ComputerCar(AbstractCar):
    IMG = GREEN_CAR
    START_POS = (488, 370)

    def __init__(self, max_vel, rotation_vel):
        super().__init__(max_vel, rotation_vel)
        self.car_mask = pygame.mask.from_surface(self.img)
        self.cumulated_rewards = 0
        self.is_finished = False
        self.is_collide = False

 
    def get_state(self):
        dx = FINISH_POSITION[0] - self.x
        dy = FINISH_POSITION[1] - self.y
        distance = math.sqrt(dx ** 2 + dy ** 2)
        angle_diff = math.atan2(dy, dx) * 180 / math.pi - self.angle
        angle_diff = (angle_diff + 180) % 360 - 180  # 限制在[-180,180]

        border_dist = self.get_distance_to_border()
        center_dist = (MID_TRACK - border_dist) / 100.0

        return np.array([
            angle_diff / 180,
            self.vel / self.max_vel,
            distance / 800,
            center_dist
        ], dtype=np.float32)
    
    def get_distance_to_border(self):
        """
        计算小车轮廓到赛道边界的最短距离（像素单位）
        """
        # 获取小车mask的所有像素点（局部坐标）
        car_mask = pygame.mask.from_surface(self.img)
        car_points = np.array(car_mask.outline())  # [(x, y), ...]
        # 转为全局坐标
        car_points_global = car_points + np.array([int(self.x), int(self.y)])
        # 交换为[y, x]以便和TRACK_BORDER_POINTS一致
        car_points_global = car_points_global[:, [1, 0]]

        # 计算所有小车边缘点到所有边界点的距离，取最小
        min_dist = np.min([
            np.min(np.linalg.norm(TRACK_BORDER_POINTS - pt, axis=1))
            for pt in car_points_global
        ])
        return min_dist
        

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
        # self.angle += np.random.uniform(-15, 15)  # 在-15到15度之间加个扰动
        return self.get_state()
    
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

        if self.collide(TRACK_BORDER_MASK):
            reward = -500  # 撞墙惩罚适当降低
            done = True
            return self.get_state(), reward, done

        if self.collide(FINISH_MASK, *FINISH_POSITION):
            reward = 3000  # 终点奖励大幅提升
            done = True
            print("FINISH!")
            print("reward:", reward)
            return self.get_state(), reward, done

        distance_delta = self.get_distance_delta()
        border_dist = self.get_distance_to_border()
        center_dist = MID_TRACK - border_dist
        print(f"Border distance: {border_dist}, Center distance: {center_dist}")

        center_reward = 100 * (1 - min(center_dist / TRACK_WIDTH, 1.0))  # 靠近中心线奖励增强


        reward += 2  # 存活奖励
        reward += 2.0 * self.vel  # 速度奖励提升
        reward += center_reward

        if distance_delta > 0.1:
            reward += 3 * distance_delta  # 鼓励移动
        else:
            reward -= 200  # 原地不动惩罚适当降低
        
        return self.get_state(), reward, done
