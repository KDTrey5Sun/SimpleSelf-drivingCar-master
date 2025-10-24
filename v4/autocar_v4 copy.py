import os
import numpy as np
import pygame
import time
import math
from draw_v4 import draw_track, WIDTH, HEIGHT, TRACK_WIDTH, LANE_WIDTH, CENTER_X, TURN_RADIUS, TURN_CENTER_OFFSET

# 赛道参数（与 draw_v4 一致）
MID_TRACK = TRACK_WIDTH / 2
RIGHT_WALL_MARGIN = 8  # 右端护栏内侧的安全余量，避免车头已越界但车心未越界的漏判

# 探索相关（基于网格的首次到访奖励、重复访问轻微惩罚）
NOVELTY_GRID = 20           # 像素网格边长
NOVELTY_BONUS = 0.8         # 首次到访奖励
REVISIT_PENALTY = 0.3       # 重复访问惩罚

# 奖励系数（缩小尺度，进度为主，其余为辅助）
CENTER_GAIN = 25.0          # 居中奖励（幅度型）
SPEED_GAIN = 0.5            # 速度线性奖励
SPEED_CENTER_SYNERGY = 0.8  # 速度×居中 协同
WRONG_WAY_COEF = 10.0       # 反向进度惩罚（progress<0）
PROGRESS_GAIN = 8.0         # 朝终点的进度奖励系数
LIVING_REWARD = 0.2         # 存活小奖励（防止过度惩罚导致早停）
DIST_DELTA_GAIN = 0.8       # 位移增益（促进流畅移动）
DIST_DELTA_NO_MOVE_PENALTY = 10.0  # 几乎无位移的小惩罚（温和）

# 速度相关提前终止
MIN_SPEED = 2.5             # 最低速度阈值（像素/帧）
LOW_SPEED_LIMIT = 60        # 低速连续步数上限
LOW_SPEED_TERM_PENALTY = 150  # 低速提前终止惩罚（降低噪声）

# 无进展提前终止（与低速独立）
NO_PROGRESS_TERM_PENALTY = 80.0

# 低速检测热身期：在前 N 步不触发低速计数，避免任务起步阶段被提前终止
LOW_SPEED_WARMUP_STEPS = 120

# 小车图片资源
def scale_image(img, factor):
    size = round(img.get_width() * factor), round(img.get_height() * factor)
    return pygame.transform.scale(img, size)

# 使用与当前文件相对的资源路径，避免工作目录变动导致找不到图片
_THIS_DIR = os.path.dirname(__file__)
_ASSETS_DIR = os.path.abspath(os.path.join(_THIS_DIR, '..', 'imgs'))

def _load_img(name):
    path = os.path.join(_ASSETS_DIR, name)
    return pygame.image.load(path)

RED_CAR = scale_image(_load_img("red-car.png"), 0.4)
GREEN_CAR = scale_image(_load_img("green-car.png"), 0.3)
CENTER_CAR = scale_image(_load_img("green-car.png"), 0.05)

FPS = 30

class AbstractCar:
    def __init__(self, max_vel, rotation_vel):
        self.START_POS = (CENTER_X, HEIGHT - 80)
        self.img = GREEN_CAR
        self.max_vel = max_vel
        self.vel = 1
        self.rotation_vel = rotation_vel
        self.angle = 0
        self.x, self.y = self.START_POS
        self.prev_x, self.prev_y = self.x, self.y  # 初始化前一位置
        self.acceleration = 0.2
        self.prev_finish_dist = None

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

    def collide_chat(self):
        # 旧的基于像素颜色的碰撞检测，保留以备调试，不在训练中使用
        win = pygame.display.get_surface()
        if win is None:
            return False
        car_rect = self.img.get_rect(center=(self.x, self.y))
        angle_rad = math.radians(self.angle)
        w, h = car_rect.width, car_rect.height
        cx, cy = self.x, self.y
        corners = []
        for dx, dy in [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]:
            rx = dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
            ry = dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
            px = int(cx + rx)
            py = int(cy + ry)
            corners.append((px, py))
        RED = (255, 0, 0)
        YELLOW = (255, 255, 0)
        BLACK = (0, 0, 0)
        for px, py in corners:
            if px < 0 or px >= WIDTH or py < 0 or py >= HEIGHT:
                return True
            color = win.get_at((px, py))[:3]
            if color in [RED, YELLOW, BLACK]:
                return True
        return False

    def collide(self):
        # 使用赛道几何（直道+圆弧+水平直线）判定是否越界
        return not self._point_in_track(self.x, self.y)

    def _point_in_track(self, x, y):
        vertical_height = HEIGHT // 2

        # 1) 垂直直道区域（去掉两侧护栏 LANE_WIDTH）
        left_border = CENTER_X - TRACK_WIDTH // 2 + LANE_WIDTH
        right_border = CENTER_X + TRACK_WIDTH // 2 - LANE_WIDTH
        if left_border <= x <= right_border and 0 <= y <= vertical_height:
            return True

        # 2) 90° 圆弧区域（角度范围约 [90°, 180°]，半径在 [inner+lane, outer-lane]）
        cx = CENTER_X + TURN_CENTER_OFFSET
        cy = vertical_height
        outer_r = TURN_RADIUS
        inner_r = TURN_RADIUS - TRACK_WIDTH
        dx = x - cx
        dy = y - cy
        dist = math.hypot(dx, dy)
        if (inner_r + LANE_WIDTH) <= dist <= (outer_r - LANE_WIDTH):
            ang = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
            if 90.0 <= ang <= 180.0:
                return True

        # 3) 水平直道区域（去掉上下护栏 LANE_WIDTH）
        horizontal_start_x = cx
        horizontal_start_y = cy + TURN_RADIUS - TRACK_WIDTH
        # 注意：右端有竖向护栏（宽度为 LANE_WIDTH），因此可行驶区域的 x 上限为 WIDTH - LANE_WIDTH。
        # 同时引入一个小余量 RIGHT_WALL_MARGIN，提前判定避免车头穿墙而车心未越界的情况。
        if (horizontal_start_x <= x <= (WIDTH - LANE_WIDTH - RIGHT_WALL_MARGIN) and
            (horizontal_start_y + LANE_WIDTH) <= y <= (horizontal_start_y + TRACK_WIDTH - 2*LANE_WIDTH)):
            return True

        return False

    def reset(self):
        self.x, self.y = self.START_POS
        self.angle = 0
        self.vel = 0
        self.prev_x, self.prev_y = self.x, self.y  # 重置时也重置 prev
        # 初始化到终点的距离
        fx, fy = CENTER_X, 40
        self.prev_finish_dist = math.hypot(fx - self.x, fy - self.y)
        # 标志位复位
        if hasattr(self, 'is_finished'):
            self.is_finished = False
        if hasattr(self, 'is_collide'):
            self.is_collide = False

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

        # 抗卡滞相关计数器
        self.no_progress_steps = 0
        self.step_count = 0
        self.max_steps = 1200  # 每回合最大步数上限（防止极端情况过长回合）

    def get_state(self):
        # 目标点为终点线中央
        finish_y = 40
        finish_x = CENTER_X
        dx = finish_x - self.x
        dy = finish_y - self.y
        distance = math.hypot(dx, dy)
        angle_diff = math.degrees(math.atan2(dy, dx)) - self.angle
        angle_diff = (angle_diff + 180) % 360 - 180  # 限制在[-180,180]

        # 使用有符号的中心偏移（左负右正），并归一化到 [-1, 1]
        center_offset = (self.x - CENTER_X) / MID_TRACK
        center_offset = max(-1.0, min(1.0, center_offset))

        return np.array([
            angle_diff / 180.0,                  # [-1, 1]
            self.vel / max(self.max_vel, 1e-6),  # [0, 1]
            min(1.5, distance / max(HEIGHT, 1e-6)),  # 大致 [0, ~1.5]
            center_offset                         # [-1, 1]
        ], dtype=np.float32)

    def get_distance_to_border(self):
        """
        几何赛道下，小车中心到“内侧护栏”的最短距离（像素）。
        针对三段赛道分别计算：
          - 垂直直道：左右内侧护栏到 x 的水平距离。
          - 圆弧：到内/外内侧半径的径向间隙。
          - 水平直道：上下内侧护栏到 y 的垂直距离。
        """
        vertical_height = HEIGHT // 2
        # 垂直段边界
        left_border = CENTER_X - TRACK_WIDTH // 2 + LANE_WIDTH
        right_border = CENTER_X + TRACK_WIDTH // 2 - LANE_WIDTH

        # 圆弧参数
        cx = CENTER_X + TURN_CENTER_OFFSET
        cy = vertical_height
        outer_r = TURN_RADIUS
        inner_r = TURN_RADIUS - TRACK_WIDTH
        inner_r_in = inner_r + LANE_WIDTH
        outer_r_in = outer_r - LANE_WIDTH

        # 水平段参数（下边界整体上移一个 LANE_WIDTH）
        horizontal_start_x = cx
        horizontal_start_y = cy + TURN_RADIUS - TRACK_WIDTH
        top_border = horizontal_start_y + LANE_WIDTH
        bottom_border = horizontal_start_y + TRACK_WIDTH - 2 * LANE_WIDTH

        x, y = self.x, self.y

        # 垂直直道区域
        if left_border <= x <= right_border and 0 <= y <= vertical_height:
            dist_left = x - left_border
            dist_right = right_border - x
            return max(0.0, min(dist_left, dist_right))

        # 圆弧区域
        dx = x - cx
        dy = y - cy
        dist = math.hypot(dx, dy)
        ang = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
        if (inner_r_in <= dist <= outer_r_in) and (90.0 <= ang <= 180.0):
            gap_inner = dist - inner_r_in
            gap_outer = outer_r_in - dist
            return max(0.0, min(gap_inner, gap_outer))

        # 水平直道区域
        if (horizontal_start_x <= x <= (WIDTH - LANE_WIDTH - RIGHT_WALL_MARGIN) and
            top_border <= y <= bottom_border):
            dist_top = y - top_border
            dist_bottom = bottom_border - y
            return max(0.0, min(dist_top, dist_bottom))

        # 不在赛道内时，返回0（避免奖励诱导）
        return 0.0

    def get_distance_delta(self):
        dx = self.x - self.prev_x
        dy = self.y - self.prev_y
        return math.sqrt(dx * dx + dy * dy)

    def reduce_speed(self):
        self.vel = max(self.vel - self.acceleration / 2, 0)
        self.move()

    def reset_env(self):
        self.reset()
        # 重置计数器
        self.no_progress_steps = 0
        self.step_count = 0
        # 每回合清空到访网格
        self.visited_bins = set()
        self.low_speed_steps = 0
        return self.get_state()

    def mark_crash_point(self):
        win = pygame.display.get_surface()
        if win is not None:
            CRASH_COLOR = (0, 0, 255)  # 蓝色
            pygame.draw.circle(win, CRASH_COLOR, (int(self.x), int(self.y)), 5)
            pygame.display.update()

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

        # 最长步数提前终止，防止极端拖延
        if self.step_count >= self.max_steps:
            done = True
            reward -= 100.0
            return self.get_state(), reward, done

        # 碰撞检测（几何边界）
        if self.collide():
            self.mark_crash_point()
            self.is_collide = True
            reward = -500.0
            done = True
            return self.get_state(), reward, done

        # 终点检测（与 draw_v4 黑白格终点线一致）
        finish_line_y = 40
        finish_line_height = 20
        start_x = CENTER_X - TRACK_WIDTH//2 + LANE_WIDTH
        end_x = CENTER_X + TRACK_WIDTH//2 - LANE_WIDTH
        car_rect = self.img.get_rect(center=(self.x, self.y))
        finish_rect = pygame.Rect(start_x, finish_line_y, end_x - start_x, finish_line_height)
        if car_rect.colliderect(finish_rect):
            reward = 3000.0
            done = True
            self.is_finished = True
            return self.get_state(), reward, done

        # --------------------------------------------------------------------------------
        # 奖励设计（进度为主，速度/居中/新奇为辅；总体尺度温和）
        # --------------------------------------------------------------------------------
        distance_delta = self.get_distance_delta()
        border_dist = self.get_distance_to_border()

        # 居中比例：内侧半宽（可行驶中心到护栏的理论值）
        inner_half = max(1.0, MID_TRACK - LANE_WIDTH)  # 防止除零
        center_ratio = max(0.0, min(1.0, border_dist / inner_half))  # [0,1]，中心≈1，靠边≈0

        # 存活 + 速度 + 居中 + 协同
        reward += LIVING_REWARD
        reward += SPEED_GAIN * (self.vel / max(self.max_vel, 1e-6))
        reward += CENTER_GAIN * center_ratio
        reward += SPEED_CENTER_SYNERGY * (self.vel / max(self.max_vel, 1e-6)) * center_ratio

        # 位移/停滞（温和）
        if distance_delta > 0.05:
            reward += DIST_DELTA_GAIN * distance_delta
        else:
            reward -= DIST_DELTA_NO_MOVE_PENALTY

        # 朝终点的真实进度
        finish_x, finish_y = CENTER_X, 40
        cur_finish_dist = math.hypot(finish_x - self.x, finish_y - self.y)
        progress = self.prev_finish_dist - cur_finish_dist  # 正：靠近终点；负：远离终点

        reward += PROGRESS_GAIN * progress
        if progress < 0:
            reward += WRONG_WAY_COEF * progress  # progress 为负数 -> 扣分（温和）

        # 无进展提前终止（阈值保持，小幅惩罚）
        if progress < 0.5:
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0
        if self.no_progress_steps >= 80:
            reward -= NO_PROGRESS_TERM_PENALTY
            done = True
            self.prev_finish_dist = cur_finish_dist
            return self.get_state(), reward, done

        # 更新终点距离
        self.prev_finish_dist = cur_finish_dist

        # 新奇度奖励/重复惩罚（弱化探索懈怠）
        bx = int(self.x // NOVELTY_GRID)
        by = int(self.y // NOVELTY_GRID)
        bkey = (bx, by)
        if not hasattr(self, 'visited_bins'):
            self.visited_bins = set()
        if bkey in self.visited_bins:
            reward -= REVISIT_PENALTY
        else:
            reward += NOVELTY_BONUS
            self.visited_bins.add(bkey)

        # 低速提前终止（热身 + 与“无进展”联动）
        if self.step_count >= LOW_SPEED_WARMUP_STEPS:
            if self.vel < MIN_SPEED:
                self.low_speed_steps += 1
            else:
                self.low_speed_steps = 0
            if self.low_speed_steps >= LOW_SPEED_LIMIT and self.no_progress_steps >= 10:
                reward -= LOW_SPEED_TERM_PENALTY
                done = True
                return self.get_state(), reward, done
        else:
            self.low_speed_steps = 0

        return self.get_state(), reward, done