import numpy as np
import pygame
import time
import math
from utils import scale_image, blit_rotate_center
from pathlib import Path

# 资源路径解析（兼容 v5_faster/imgs 与 仓库根/imgs）
ASSET_DIRS = [
    Path(__file__).resolve().parent / "imgs",
    Path(__file__).resolve().parent.parent / "imgs",
]
def asset(name: str) -> str:
    for d in ASSET_DIRS:
        p = d / name
        if p.exists():
            return str(p.resolve())
    raise FileNotFoundError(f"Asset not found: {name}; tried: {[str(d) for d in ASSET_DIRS]}")

# 加载图像资源（绝对路径）
GRASS = scale_image(pygame.image.load(asset("grass.jpg")), 2.5)
TRACK = scale_image(pygame.image.load(asset("track1.png")), 0.3)
TRACK_BORDER = scale_image(pygame.image.load(asset("track_border1.png")), 0.3)
TRACK_BORDER_MASK = pygame.mask.from_surface(TRACK_BORDER)
FINISH = pygame.image.load(asset("finish.png"))
FINISH_MASK = pygame.mask.from_surface(FINISH)
FINISH_POSITION = (453, 410)
RED_CAR = scale_image(pygame.image.load(asset("red-car.png")), 0.4)
GREEN_CAR = scale_image(pygame.image.load(asset("green-car.png")), 0.3)
CENTER_CAR = scale_image(pygame.image.load(asset("green-car.png")), 0.05)

# 设置窗口
WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Racing Game!")
FPS = 30

TRACK_WIDTH = 70
MID_TRACK = TRACK_WIDTH / 2
CONST_SPEED = 3

# 奖励/终止相关
SURVIVE_REWARD = 0.2
PROGRESS_GAIN = 10.0
CENTER_GAIN = 12.0
ALIGN_GAIN = 8.0
PROGRESS_THRESH = 0.003
NO_PROGRESS_LIMIT = 120
NO_PROGRESS_TERM_PENALTY = 300.0
CURVE_DETECT_THRESH = 0.012

# 赛道中心
RING_CX = WIDTH / 2.0
RING_CY = HEIGHT / 2.0

# 预计算：边界的欧式距离变换（到最近边界像素的距离，单位像素）
DIST_MAP = None
try:
    from scipy.ndimage import distance_transform_edt
    arr = pygame.surfarray.array3d(TRACK_BORDER)   # (W,H,3)
    arr = np.transpose(arr, (1, 0, 2))             # -> (H,W,3)
    border_mask = np.any(arr != 0, axis=2)         # True 表示边界像素
    non_border = ~border_mask
    DIST_MAP = distance_transform_edt(non_border).astype(np.float32)  # (H,W)
except Exception as e:
    DIST_MAP = None
    print(f"[WARN] DIST_MAP unavailable, fallback to slow distance calc: {e}")

# 慢速回退所需：赛道边界点集（惰性构建）
_TRACK_BORDER_POINTS = None
def _get_track_border_points(surface):
    global _TRACK_BORDER_POINTS
    if _TRACK_BORDER_POINTS is None:
        arr = pygame.surfarray.array3d(surface)
        arr = np.transpose(arr, (1, 0, 2))
        mask = np.any(arr != 0, axis=2)
        _TRACK_BORDER_POINTS = np.argwhere(mask)  # [y,x]
    return _TRACK_BORDER_POINTS

# 旋转与掩码缓存：key=(id(img), angle_int)
_ROT_CACHE = {}
def get_rotated_cached(img: pygame.Surface, angle_deg: float):
    akey = int(round(angle_deg)) % 360
    key = (id(img), akey)
    hit = _ROT_CACHE.get(key)
    if hit is not None:
        return hit
    rotated = pygame.transform.rotate(img, akey)
    mask = pygame.mask.from_surface(rotated)
    _ROT_CACHE[key] = (rotated, mask)
    return rotated, mask

class AbstractCar:
    IMG = None
    def __init__(self, max_vel, rotation_vel):
        self.START_POS = (488, 370)
        if not hasattr(self, 'IMG') or self.IMG is None:
            raise AttributeError("Subclass must define class attribute 'IMG'")
        self.img = self.IMG
        self.max_vel = max_vel
        self.vel = CONST_SPEED
        self.rotation_vel = rotation_vel
        self.angle = 0
        self.x, self.y = self.START_POS
        self.prev_x, self.prev_y = self.x, self.y
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
        self.vel = CONST_SPEED
        self.move()

    def move_backward(self):
        self.vel = CONST_SPEED
        self.move()

    def move(self):
        radians = math.radians(self.angle)
        vertical = math.cos(radians) * self.vel
        horizontal = math.sin(radians) * self.vel
        self.y -= vertical
        self.x -= horizontal

    def collide(self, mask, x=0, y=0):
        rotated, car_mask = get_rotated_cached(self.img, self.angle)
        car_rect = rotated.get_rect(center=(self.x, self.y))
        offset = (int(car_rect.left - x), int(car_rect.top - y))
        return mask.overlap(car_mask, offset)

    def reset(self):
        self.x, self.y = self.START_POS
        self.angle = 0
        self.vel = CONST_SPEED
        self.prev_x, self.prev_y = self.x, self.y
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
        self.no_progress_steps = 0
        self.step_count = 0
        self.max_steps = 1200
        self.prev_dir_x = 0.0
        self.prev_dir_y = -1.0

    def get_state(self):
        theta_pos = math.atan2(RING_CY - self.y, self.x - RING_CX)
        heading = math.radians(self.angle)
        hx, hy = -math.sin(heading), -math.cos(heading)
        tx, ty = -math.sin(theta_pos), -math.cos(theta_pos)
        dot = max(-1.0, min(1.0, hx * tx + hy * ty))
        cross_z = max(-1.0, min(1.0, hx * ty - hy * tx))
        border_dist = self.get_distance_to_border()
        denom = max(MID_TRACK, 1e-6)
        center_delta = (MID_TRACK - border_dist) / denom
        center_delta = max(-1.0, min(1.0, center_delta))
        return np.array([dot, cross_z, center_delta, math.cos(theta_pos), math.sin(theta_pos)], dtype=np.float32)

    def get_distance_to_border(self):
        """
        小车轮廓到赛道边界的最短距离（像素）。
        优先使用预计算的 DIST_MAP；不可用时回退到慢速算法（子采样）。
        """
        rotated, car_mask = get_rotated_cached(self.img, self.angle)
        outline = car_mask.outline()
        if not outline:
            return 0.0
        car_rect = rotated.get_rect(center=(self.x, self.y))
        pts = np.asarray(outline, dtype=np.int32)
        ys = pts[:, 1] + int(car_rect.top)
        xs = pts[:, 0] + int(car_rect.left)

        if DIST_MAP is not None:
            H, W = DIST_MAP.shape
            ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
            if not np.any(ok):
                return 0.0
            xs_ok, ys_ok = xs[ok], ys[ok]
            return float(np.min(DIST_MAP[ys_ok, xs_ok]))

        # 慢速回退：子采样轮廓点减少计算量
        if len(xs) > 128:
            step = max(1, len(xs) // 128)
            xs = xs[::step]; ys = ys[::step]
        car_points_global = np.stack([ys, xs], axis=1)  # [y,x]
        border_pts = _get_track_border_points(TRACK_BORDER)
        dmin = min(np.min(np.linalg.norm(border_pts - pt, axis=1)) for pt in car_points_global)
        return float(dmin)

    def reduce_speed(self):
        self.vel = CONST_SPEED
        self.move()

    def reset_env(self):
        self.reset()
        self.no_progress_steps = 0
        self.step_count = 0
        self.prev_dir_x = 0.0
        self.prev_dir_y = -1.0
        return self.get_state()

    def step(self, action):
        self.prev_x, self.prev_y = self.x, self.y
        self.step_count += 1

        if action == 0:
            self.rotate(left=True);  self.move_forward()
        elif action == 1:
            self.rotate(right=True); self.move_forward()
        elif action == 2:
            self.move_forward()
        elif action == 3:
            self.reduce_speed()

        done = False
        reward = 0.0

        if self.step_count >= self.max_steps:
            done = True; reward -= 100.0; self.termination_reason = 'max_steps'
            return self.get_state(), reward, done

        if self.collide(TRACK_BORDER_MASK):
            self.is_collide = True; reward = -500; done = True; self.termination_reason = 'collision'
            return self.get_state(), reward, done

        if self.collide(FINISH_MASK, *FINISH_POSITION):
            reward = 3000; done = True; self.is_finished = True; self.termination_reason = 'finish'
            return self.get_state(), reward, done

        dx = self.x - self.prev_x
        dy = self.y - self.prev_y
        theta_prev = math.atan2(RING_CY - self.prev_y, self.prev_x - RING_CX)
        theta_cur  = math.atan2(RING_CY - self.y,      self.x      - RING_CX)
        dtheta = (theta_cur - theta_prev + math.pi) % (2 * math.pi) - math.pi
        angle_progress = max(0.0, dtheta)
        progress_term = PROGRESS_GAIN * angle_progress

        border_dist = self.get_distance_to_border()
        center_offset = abs(MID_TRACK - border_dist)
        center_factor = 1.0 - min(1.0, center_offset / max(MID_TRACK, 1e-6))
        center_term = CENTER_GAIN * center_factor

        tx, ty = -math.sin(theta_cur), -math.cos(theta_cur)
        step_norm = math.hypot(dx, dy)
        if step_norm < 1e-6:
            heading = math.radians(self.angle)
            sx, sy = -math.sin(heading), -math.cos(heading)
        else:
            sx, sy = dx / step_norm, dy / step_norm
        pdx, pdy = self.prev_dir_x, self.prev_dir_y
        w_curve = min(1.0, abs(dtheta) / CURVE_DETECT_THRESH)
        align_curve = abs(sx * tx + sy * ty)
        align_straight = abs(sx * pdx + sy * pdy)
        align_factor = w_curve * align_curve + (1.0 - w_curve) * align_straight
        align_term = ALIGN_GAIN * align_factor

        reward += SURVIVE_REWARD + progress_term + center_term + align_term

        if angle_progress < PROGRESS_THRESH:
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0
        if self.no_progress_steps >= NO_PROGRESS_LIMIT:
            reward -= NO_PROGRESS_TERM_PENALTY
            done = True
            self.termination_reason = 'no_progress'
            return self.get_state(), reward, done

        self.prev_dir_x, self.prev_dir_y = sx, sy
        return self.get_state(), reward, done