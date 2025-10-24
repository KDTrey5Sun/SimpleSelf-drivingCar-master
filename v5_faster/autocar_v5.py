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

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Racing Game!")
FPS = 30

TRACK_WIDTH = 70
MID_TRACK = TRACK_WIDTH / 2

# 固定速度（像素/步）
CONST_SPEED = 3

# 轻量生存 + 终点进度奖励（相对终点的距离减少）
SURVIVE_REWARD = 0.2
PROGRESS_GAIN = 10.0  # 角度奖励

# 居中与切向对齐奖励（圆形赛道更稳定）
CENTER_GAIN = 12.0
ALIGN_GAIN = 8.0

# 无进展提前终止（角度制，单位：弧度/步）
# 稍微放宽阈值，减少误杀：更小的阈值 + 更长的等待步数
PROGRESS_THRESH = 0.003  # ~0.17°/步
NO_PROGRESS_LIMIT = 120
NO_PROGRESS_TERM_PENALTY = 300.0

# 弯直段判别阈值（按每步角度变化衡量曲率），超过该阈值认为在弯道
CURVE_DETECT_THRESH = 0.012  # rad/step，约 0.7°/步

def get_track_border_points(surface):
    arr = pygame.surfarray.array3d(surface)
    arr = np.transpose(arr, (1, 0, 2))
    mask = np.any(arr != 0, axis=2)
    points = np.argwhere(mask)
    return points
TRACK_BORDER_POINTS = get_track_border_points(TRACK_BORDER)

# 赛道中心点直接取图片中心（用户指定）：
RING_CX = WIDTH / 2.0
RING_CY = HEIGHT / 2.0

class AbstractCar:
    # 在基类上声明类属性，便于静态检查；子类应赋为 pygame.Surface
    IMG = None
    def __init__(self, max_vel, rotation_vel):
        self.START_POS = (488, 370)
        # 允许子类定义 IMG；为基类提供兜底与清晰的错误提示
        if not hasattr(self, 'IMG') or self.IMG is None:
            raise AttributeError("Subclass must define class attribute 'IMG' before AbstractCar.__init__")
        self.img = self.IMG
        self.max_vel = max_vel
        # 固定速度初始化
        self.vel = CONST_SPEED
        self.rotation_vel = rotation_vel
        self.angle = 0
        self.x, self.y = self.START_POS
        self.prev_x, self.prev_y = self.x, self.y  # 初始化前一位置
        # 禁用加速度（固定速度）
        self.acceleration = 0.0
        self.prev_finish_dist = None
        self.termination_reason = None

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
        # 固定速度前进
        self.vel = CONST_SPEED
        self.move()

    def move_backward(self):
        # 恒速模式下，不提供倒退，加速/减速动作退化为恒速前进
        self.vel = CONST_SPEED
        self.move()

    def move(self):
        radians = math.radians(self.angle)
        vertical = math.cos(radians) * self.vel
        horizontal = math.sin(radians) * self.vel
        self.y -= vertical
        self.x -= horizontal

    def collide(self, mask, x=0, y=0):
        """与给定掩码做像素级碰撞检测。
        - mask: 赛道边界/终点的 pygame.mask.Mask
        - x,y: 该掩码在窗口上的左上角坐标（边界图通常为 0,0；终点线为 FINISH_POSITION）
        计算方式：旋转车身图，按中心放置到 (self.x,self.y)，取其 rect 与掩码做 overlap。
        """
        rotated = pygame.transform.rotate(self.img, self.angle)
        car_mask = pygame.mask.from_surface(rotated)
        car_rect = rotated.get_rect(center=(self.x, self.y))
        # othermask(车)相对mask(边界/终点)的偏移
        offset = (int(car_rect.left - x), int(car_rect.top - y))
        return mask.overlap(car_mask, offset)

    def reset(self):
        self.x, self.y = self.START_POS
        self.angle = 0
        self.vel = CONST_SPEED
        self.prev_x, self.prev_y = self.x, self.y  # 重置时也重置 prev
        # 初始化到终点中心的距离（用于进度奖励）
        fx = FINISH_POSITION[0] + FINISH.get_width() / 2.0
        fy = FINISH_POSITION[1] + FINISH.get_height() / 2.0
        self.prev_finish_dist = math.hypot(fx - self.x, fy - self.y)
        self.termination_reason = None


class ComputerCar(AbstractCar):
    IMG = GREEN_CAR
    START_POS = (488, 370)

    def __init__(self, max_vel, rotation_vel):
        super().__init__(max_vel, rotation_vel)
        self.car_mask = pygame.mask.from_surface(self.img)
        self.cumulated_rewards = 0
        self.is_finished = False
        self.is_collide = False

        # 抗卡滞相关计数器
        self.no_progress_steps = 0
        self.step_count = 0
        self.max_steps = 1200
        # 上一步的运动方向（屏幕坐标），用于直线段对齐奖励
        self.prev_dir_x = 0.0
        self.prev_dir_y = -1.0

 
    def get_state(self):
        """基于赛道圆心/切向的观测，去除与终点相关的量。
        返回 5 维特征：
        [cos(delta_heading_tangent), sin(delta_heading_tangent), center_delta, cos(theta_pos), sin(theta_pos)]
        其中 delta_heading_tangent = 车头朝向(弧度) - 切向角(弧度)，theta_pos 是圆心坐标系下的位置角。
        center_delta = (MID_TRACK - border_dist) / MID_TRACK，范围约 [-1,1]。
        """
        # 位置角（相对圆心，使用数学坐标：y 向上为正，这样逆时针为角度正向）
        theta_pos = math.atan2(RING_CY - self.y, self.x - RING_CX)
        # 车头朝向（度->弧度），0=朝上，角度增大为左转（屏幕坐标系下的逆时针）
        heading = math.radians(self.angle)
        # 使用方向向量的点积/叉积直接得到 cos(delta)/sin(delta)，避免角度系转换误差
        # 车头前向（屏幕坐标）
        hx, hy = -math.sin(heading), -math.cos(heading)
        # CCW 切向单位向量（屏幕坐标）：基于数学角度 theta_pos 转换到屏幕方向
        tx, ty = -math.sin(theta_pos), -math.cos(theta_pos)
        # 单位长度下：dot = cos(delta), cross_z = sin(delta)
        dot = hx * tx + hy * ty
        cross_z = hx * ty - hy * tx
        # 数值稳定：夹紧
        dot = max(-1.0, min(1.0, dot))
        cross_z = max(-1.0, min(1.0, cross_z))
        cos_d, sin_d = dot, cross_z

        # 居中偏差（相对赛道中线，带符号，正值表示偏向内/外侧取决于定义）
        border_dist = self.get_distance_to_border()
        denom = max(MID_TRACK, 1e-6)
        center_delta = (MID_TRACK - border_dist) / denom
        # 限幅
        center_delta = max(-1.0, min(1.0, center_delta))

        return np.array([
            cos_d,
            sin_d,
            center_delta,
            math.cos(theta_pos),
            math.sin(theta_pos),
        ], dtype=np.float32)
    
    def get_distance_to_border(self):
        """
        计算小车轮廓到赛道边界的最短距离（像素单位）
        """
        # 获取旋转后的小车 mask 的外轮廓点（局部坐标）
        rotated = pygame.transform.rotate(self.img, self.angle)
        car_mask = pygame.mask.from_surface(rotated)
        car_points = np.array(car_mask.outline())  # [(x, y), ...]
        # 转为全局坐标
        car_rect = rotated.get_rect(center=(self.x, self.y))
        car_points_global = car_points + np.array([int(car_rect.left), int(car_rect.top)])
        # 交换为[y, x]以便和TRACK_BORDER_POINTS一致
        car_points_global = car_points_global[:, [1, 0]]

        # 计算所有小车边缘点到所有边界点的距离，取最小
        min_dist = min(
            np.min(np.linalg.norm(TRACK_BORDER_POINTS - pt, axis=1))
            for pt in car_points_global
        ) if len(car_points_global) else 0.0
        return float(min_dist)

    def reduce_speed(self):
        # 恒速：减速动作退化为恒速前进
        self.vel = CONST_SPEED
        self.move()
    
    def reset_env(self):
        self.reset()
        self.no_progress_steps = 0
        self.step_count = 0
        self.prev_dir_x = 0.0
        self.prev_dir_y = -1.0
        # self.angle += np.random.uniform(-15, 15)  # 在-15到15度之间加个扰动
        return self.get_state()
    
    def step(self, action):
        self.prev_x, self.prev_y = self.x, self.y
        # 步数统计与上限终止
        self.step_count += 1

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
        reward = 0.0

        # 最长步数提前终止
        if self.step_count >= self.max_steps:
            done = True
            reward -= 100.0
            self.termination_reason = 'max_steps'
            return self.get_state(), reward, done

        # 碰撞检测（赛道边界）
        if self.collide(TRACK_BORDER_MASK):
            self.is_collide = True
            reward = -500
            done = True
            self.termination_reason = 'collision'
            return self.get_state(), reward, done

        # 终点线检测
        if self.collide(FINISH_MASK, *FINISH_POSITION):
            reward = 3000
            done = True
            self.is_finished = True
            self.termination_reason = 'finish'
            return self.get_state(), reward, done

        # 奖励：圆心角进度（只奖励角度增大） + 居中 + 对齐 + 生存
        # 1) 圆心角进度：以图片中心为圆心，θ = atan2(y - RING_CY, x - RING_CX)
        dx = self.x - self.prev_x
        dy = self.y - self.prev_y
        # 使用数学坐标系角度（y 向上）确保逆时针为正方向
        theta_prev = math.atan2(RING_CY - self.prev_y, self.prev_x - RING_CX)
        theta_cur = math.atan2(RING_CY - self.y, self.x - RING_CX)
        dtheta = (theta_cur - theta_prev + math.pi) % (2 * math.pi) - math.pi  # wrap 到 (-pi, pi]
        angle_progress = max(0.0, dtheta)  # 仅奖励“角度增大”的部分（只允许 CCW）
        progress_term = PROGRESS_GAIN * angle_progress

        # 2) 居中奖励：使用边界最近距离与 MID_TRACK 的偏差
        border_dist = self.get_distance_to_border()
        center_offset = abs(MID_TRACK - border_dist)
        center_factor = 1.0 - min(1.0, center_offset / max(MID_TRACK, 1e-6))
        center_term = CENTER_GAIN * center_factor

        # 3) 对齐奖励（弯-直自适应）：
        # 弯道：与切向一致；直线：与上一步方向一致；按曲率权重平滑过渡
        theta = theta_cur
        # CCW 切向单位向量（屏幕坐标）
        tx, ty = -math.sin(theta), -math.cos(theta)
        step_norm = math.hypot(dx, dy)
        if step_norm < 1e-6:
            # 步进很小，用车头朝向代替
            heading = math.radians(self.angle)
            sx, sy = -math.sin(heading), -math.cos(heading)
            step_norm = 1.0
        else:
            sx, sy = dx / step_norm, dy / step_norm
        # 上一步方向单位向量
        pdx, pdy = self.prev_dir_x, self.prev_dir_y
        # 曲率权重（弯道越大权重越高）
        w_curve = min(1.0, abs(dtheta) / CURVE_DETECT_THRESH)
        # 两种对齐度
        align_curve = abs(sx * tx + sy * ty)
        align_straight = abs(sx * pdx + sy * pdy)
        align_factor = w_curve * align_curve + (1.0 - w_curve) * align_straight
        align_term = ALIGN_GAIN * align_factor

        reward += SURVIVE_REWARD + progress_term + center_term + align_term
        # 无进展提前终止：使用角度进度判断
        if angle_progress < PROGRESS_THRESH:
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0
        if self.no_progress_steps >= NO_PROGRESS_LIMIT:
            reward -= NO_PROGRESS_TERM_PENALTY
            done = True
            self.termination_reason = 'no_progress'
            return self.get_state(), reward, done

        # 更新 prev_dir（用于下一步的直线对齐）
        self.prev_dir_x, self.prev_dir_y = sx, sy
        return self.get_state(), reward, done
