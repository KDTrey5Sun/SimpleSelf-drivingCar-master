import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
pygame.init()

import sys
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from v5.autocar_v5 import ComputerCar, _ensure_border_mask_loaded, _BORDER_MASK

# 创建最小窗口以满足某些 API 要求
pygame.display.set_mode((1, 1))

# 触发掩码加载
_ensure_border_mask_loaded()
print("border_mask_loaded:", _BORDER_MASK is not None)

car = ComputerCar(max_vel=8, rotation_vel=4)
state = car.reset_env()
print("initial_state:", state)

# 执行几步前进并打印是否碰撞
for step in range(5):
    s, r, done = car.step(2)  # 2 = forward
    print(f"step={step} x={car.x:.1f} y={car.y:.1f} vel={car.vel:.2f} r={r:.1f} done={done}")
    if done:
        break

pygame.quit()
print("smoke_test_done")
