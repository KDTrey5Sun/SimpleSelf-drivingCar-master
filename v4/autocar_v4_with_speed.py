import numpy as np
import pygame
import math
from draw_v4 import WIDTH, HEIGHT, TRACK_WIDTH, LANE_WIDTH, CENTER_X, TURN_RADIUS, TURN_CENTER_OFFSET

# 赛道参数（与 draw_v4 一致）
MID_TRACK = TRACK_WIDTH / 2 # 赛道中心线到任一侧护栏的距离

RIGHT_WALL_MARGIN = 8  # 右端护栏内侧的安全余量，避免车头已越界但车心未越界的漏判


"""
极简奖励机制：
- 生存奖励：每步一个小正数，鼓励持续探索
- 进度奖励：朝终点距离的减少量（progress）的线性奖励
保留的终止条件：碰撞、到达终点、最大步数、长时间无进展。
"""

# 奖励权重（专注：居中、方向正确、保持速度）
SURVIVE_REWARD = 0.2        # 轻量生存
PROGRESS_GAIN = 3           # 轻量进度（避免主导）
CENTER_GAIN = 60            # 居中奖励（越居中越高）
ALIGN_GAIN = 40             # 朝向对齐奖励（越对准终点越高）
SPEED_GAIN = 1              # 速度奖励（高于阈值线性加分）
SPEED_MIN = 1               # 建议的最低速度

FPS = 30

# 无进展判定与上限
PROGRESS_THRESH = 0.25   # 低于该进度视为“无进展”
NO_PROGRESS_LIMIT = 120  # 连续无进展步数上限
NO_PROGRESS_TERM_PENALTY = 150  # 终止时的额外小惩罚


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

GREEN_CAR = scale_image(_load_image("imgs/green-car.png", (0, 200, 0)), 0.3)


class AbstractCar:
    # 在基类上声明类属性，便于静态检查；子类应赋为 pygame.Surface
    IMG = None
    def __init__(self, max_vel, rotation_vel):
        self.START_POS = (CENTER_X, HEIGHT - 80)
        # 允许子类定义 IMG；为基类提供兜底与清晰的错误提示
        if not hasattr(self, 'IMG') or self.IMG is None:
            raise AttributeError("Subclass must define class attribute 'IMG' before AbstractCar.__init__")
        self.img = self.IMG
        self.max_vel = max_vel
        self.vel = 1
        self.rotation_vel = rotation_vel
        self.angle = 0
        self.x, self.y = self.START_POS
        self.acceleration = 0.2
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
        self.vel = min(self.vel + self.acceleration, self.max_vel)
        self.move()

    def move(self):
        radians = math.radians(self.angle)
        vertical = math.cos(radians) * self.vel
        horizontal = math.sin(radians) * self.vel
        self.y -= vertical
        self.x -= horizontal

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
        # 初始化到终点的距离
        fx, fy = CENTER_X, 40
        self.prev_finish_dist = math.hypot(fx - self.x, fy - self.y)
        # 标志位复位
        if hasattr(self, 'is_finished'):
            self.is_finished = False
        if hasattr(self, 'is_collide'):
            self.is_collide = False
        self.termination_reason = None

def blit_rotate_center(win, image, top_left, angle):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center=image.get_rect(topleft=top_left).center)
    win.blit(rotated_image, new_rect.topleft)

class ComputerCar(AbstractCar):
    IMG = GREEN_CAR
    START_SHIFT_X = 200
    base_start_x = CENTER_X + 400
    base_start_y = HEIGHT // 2 + TURN_RADIUS - TRACK_WIDTH // 2
    START_POS = (base_start_x + START_SHIFT_X, base_start_y)

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

    # 删除了未使用的几何距离与位移计算以精简代码

    def reduce_speed(self):
        self.vel = max(self.vel - self.acceleration / 2, 0)
        self.move()

    def reset_env(self):
        self.reset()
        # 重置计数器
        self.no_progress_steps = 0
        self.step_count = 0
        return self.get_state()

    def step(self, action):
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
            self.termination_reason = 'max_steps'
            return self.get_state(), reward, done


        # 碰撞检测（几何边界）
        if self.collide():
            self.is_collide = True
            reward = -500
            done = True
            self.termination_reason = 'collision'
            return self.get_state(), reward, done

        # 终点检测（与 draw_v4 黑白格终点线一致）
        finish_line_y = 40
        finish_line_height = 20
        start_x = CENTER_X - TRACK_WIDTH//2 + LANE_WIDTH
        end_x = CENTER_X + TRACK_WIDTH//2 - LANE_WIDTH
        car_rect = self.img.get_rect(center=(self.x, self.y))
        finish_rect = pygame.Rect(start_x, finish_line_y, end_x - start_x, finish_line_height)
        if car_rect.colliderect(finish_rect):
            reward = 3000
            done = True
            self.is_finished = True
            self.termination_reason = 'finish'
            return self.get_state(), reward, done

        # 奖励：居中 + 对齐 + 速度 + 轻量进度 + 生存
        finish_x, finish_y = CENTER_X, 40

        # 判断所在赛道段并计算中心线偏差与期望朝向
        vertical_height = HEIGHT // 2
        cx = CENTER_X + TURN_CENTER_OFFSET
        cy = vertical_height
        outer_r = TURN_RADIUS
        inner_r = TURN_RADIUS - TRACK_WIDTH
        # 直道与弯道的边界（与 _point_in_track 保持一致）
        left_border = CENTER_X - TRACK_WIDTH // 2 + LANE_WIDTH
        right_border = CENTER_X + TRACK_WIDTH // 2 - LANE_WIDTH
        horizontal_start_x = cx
        horizontal_start_y = cy + TURN_RADIUS - TRACK_WIDTH
        top_border = horizontal_start_y + LANE_WIDTH
        bottom_border = horizontal_start_y + TRACK_WIDTH - 2 * LANE_WIDTH

        seg = 'unknown'
        desired_angle_deg = 0.0
        # 默认以垂直段规则计算中心偏差
        center_offset_norm = abs(self.x - CENTER_X) / max(MID_TRACK, 1e-6)

        if left_border <= self.x <= right_border and 0 <= self.y <= vertical_height:
            seg = 'vertical'
            desired_angle_deg = 0.0  # 向上
            center_offset_norm = abs(self.x - CENTER_X) / max(MID_TRACK, 1e-6)
        else:
            dx_c = self.x - cx
            dy_c = self.y - cy
            dist_c = math.hypot(dx_c, dy_c)
            ang = (math.degrees(math.atan2(dy_c, dx_c)) + 360.0) % 360.0
            if (inner_r + LANE_WIDTH) <= dist_c <= (outer_r - LANE_WIDTH) and 90.0 <= ang <= 180.0:
                seg = 'arc'
                # 中心线半径（几何居中）
                r_center = TURN_RADIUS - TRACK_WIDTH / 2.0
                center_offset_norm = abs(dist_c - r_center) / max(MID_TRACK, 1e-6)
                # 弧线切线方向（沿左转方向），切向向量 t = (-dy, dx)
                tx, ty = -dy_c, dx_c
                norm = math.hypot(tx, ty) or 1.0
                tx, ty = tx / norm, ty / norm
                desired_angle_deg = (math.degrees(math.atan2(-tx, -ty)))  # 由速度映射反推角度
            elif (horizontal_start_x <= self.x <= (WIDTH - LANE_WIDTH - RIGHT_WALL_MARGIN) and
                  top_border <= self.y <= bottom_border):
                seg = 'horizontal'
                desired_angle_deg = 90.0  # 向左
                center_y = (top_border + bottom_border) * 0.5
                # 使用垂直偏差作为“居中”，归一化到与直道一致的尺度
                center_offset_norm = abs(self.y - center_y) / max(MID_TRACK, 1e-6)

        center_reward = CENTER_GAIN * (1.0 - min(1.0, center_offset_norm))

        # 朝路径方向对齐（与期望角度的差值）
        angle_diff = (desired_angle_deg - self.angle + 180) % 360 - 180
        align_factor = max(0.0, 1.0 - abs(angle_diff) / 90.0)
        align_reward = ALIGN_GAIN * align_factor

        # 速度：高于最低速度阈值的部分加分
        speed_reward = SPEED_GAIN * max(0.0, self.vel - SPEED_MIN)

        # 轻量进度
        cur_finish_dist = math.hypot(finish_x - self.x, finish_y - self.y)
        progress = self.prev_finish_dist - cur_finish_dist
        progress_reward = PROGRESS_GAIN * progress

        reward += center_reward + align_reward + speed_reward + progress_reward + SURVIVE_REWARD

        # 无进展提前终止（不额外惩罚反向，进度为负本身会扣分）
        if progress < PROGRESS_THRESH:
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0
        if self.no_progress_steps >= NO_PROGRESS_LIMIT:
            reward -= NO_PROGRESS_TERM_PENALTY
            done = True
            self.prev_finish_dist = cur_finish_dist
            self.termination_reason = 'no_progress'
            return self.get_state(), reward, done

        # 更新终点距离
        self.prev_finish_dist = cur_finish_dist

        return self.get_state(), reward, done


