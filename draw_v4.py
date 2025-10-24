import pygame
import math

# 画布和赛道参数
WIDTH, HEIGHT = 1516, 800
TRACK_WIDTH = 200
LANE_WIDTH = 20
CENTER_X = WIDTH // 2
TURN_RADIUS = 200  # 弯道外圈半径
TURN_CENTER_OFFSET = TURN_RADIUS - TRACK_WIDTH // 2  # 圆心相对直线中心的偏移
TURN_ARC_ANGLE = 90  # 90度转弯

def draw_ring_sector(win, color, center, r_outer, r_inner, angle_start, angle_end, step=1):
    # 画圆环扇形（用于填充弯道赛道，彻底盖住黑点）
    points = []
    # 外圈
    for a in range(angle_start, angle_end+1, step):
        rad = math.radians(a)
        x = center[0] + r_outer * math.cos(rad)
        y = center[1] + r_outer * math.sin(rad)
        points.append((x, y))
    # 内圈（反向）
    for a in range(angle_end, angle_start-1, -step):
        rad = math.radians(a)
        x = center[0] + r_inner * math.cos(rad)
        y = center[1] + r_inner * math.sin(rad)
        points.append((x, y))
    pygame.draw.polygon(win, color, points)

def draw_track(win):
    BLACK = (0, 0, 0)
    GRAY = (100, 100, 100)
    RED = (255, 0, 0)
    YELLOW = (255, 255, 0)
    WHITE = (255, 255, 255)

    win.fill(BLACK)

    vertical_height = HEIGHT // 2

    # -------- 垂直部分 --------
 
    pygame.draw.rect(win, GRAY, (CENTER_X - TRACK_WIDTH//2, 0,
                                 TRACK_WIDTH, vertical_height))
    for i in range(0, vertical_height, 40):
        color = RED if (i // 40) % 2 == 0 else YELLOW
        pygame.draw.rect(win, color, (CENTER_X - TRACK_WIDTH//2, i, LANE_WIDTH, 40))
        pygame.draw.rect(win, color, (CENTER_X + TRACK_WIDTH//2 - LANE_WIDTH, i, LANE_WIDTH, 40))

    # -------- 圆弧弯道部分（用多边形填充灰色赛道和绿色内道） --------
    arc_center_x = CENTER_X + TURN_CENTER_OFFSET
    arc_center_y = vertical_height
    outer_radius = TURN_RADIUS
    inner_radius = TURN_RADIUS - TRACK_WIDTH
    arc_radius_inner = TURN_RADIUS - TRACK_WIDTH + LANE_WIDTH

    # 灰色赛道圆环填充（彻底盖住黑点）
    draw_ring_sector(win, GRAY, (arc_center_x, arc_center_y),outer_radius, inner_radius, 90, 180, step=1)


    # 内侧红黄护栏
    arc_rect_inner = [
        arc_center_x - arc_radius_inner,
        arc_center_y - arc_radius_inner,
        2 * arc_radius_inner,
        2 * arc_radius_inner
    ]
    arc_len_inner = int(arc_radius_inner * math.radians(TURN_ARC_ANGLE))
    for i in range(0, arc_len_inner, 40):
        color = RED if (i // 40) % 2 == 0 else YELLOW
        angle1 = math.radians(180) + (math.radians(90) * (i / arc_len_inner))
        angle2 = math.radians(180) + (math.radians(90) * ((i + 40) / arc_len_inner))
        pygame.draw.arc(win, color, arc_rect_inner, angle1, min(angle2, math.radians(270)), LANE_WIDTH)

    # 外侧红黄护栏
    arc_rect_outer = [
        arc_center_x - outer_radius,
        arc_center_y - outer_radius,
        2 * outer_radius,
        2 * outer_radius
    ]
    arc_len_outer = int(outer_radius * math.radians(TURN_ARC_ANGLE))
    for i in range(0, arc_len_outer, 40):
        color = RED if (i // 40) % 2 == 0 else YELLOW
        angle1 = math.radians(180) + (math.radians(90) * (i / arc_len_outer))
        angle2 = math.radians(180) + (math.radians(90) * ((i + 40) / arc_len_outer))
        pygame.draw.arc(win, color, arc_rect_outer, angle1, min(angle2, math.radians(270)), LANE_WIDTH)

    # -------- 水平直线部分（仅右侧延长到窗口边缘） --------
    horizontal_start_x = arc_center_x
    horizontal_start_y = arc_center_y + TURN_RADIUS - TRACK_WIDTH

    # 灰色赛道（右侧延长到窗口边缘）
    pygame.draw.rect(win, GRAY, (horizontal_start_x, horizontal_start_y,
                                 WIDTH - horizontal_start_x, TRACK_WIDTH))

    # 上红黄护栏（右侧延长部分）
    for i in range(0, WIDTH - horizontal_start_x, 40):
        color = RED if (i // 40) % 2 == 0 else YELLOW
        pygame.draw.rect(win, color, (horizontal_start_x + i, horizontal_start_y, 40, LANE_WIDTH))
    # 下红黄护栏（右侧延长部分）
    for i in range(0, WIDTH - horizontal_start_x, 40):
        color = RED if (i // 40) % 2 == 0 else YELLOW
        pygame.draw.rect(win, color, (horizontal_start_x + i, horizontal_start_y + TRACK_WIDTH - LANE_WIDTH, 40, LANE_WIDTH))

    # -------- 水平赛道右端封口红黄护栏 --------
    # 上护栏
    for j in range(0, TRACK_WIDTH, 40):
        color = RED if (j // 40) % 2 == 0 else YELLOW
        pygame.draw.rect(win, color, (WIDTH - 40, horizontal_start_y + j, 40, LANE_WIDTH))
    # 下护栏
    for j in range(0, TRACK_WIDTH, 40):
        color = RED if (j // 40) % 2 == 0 else YELLOW
        pygame.draw.rect(win, color, (WIDTH - 40, horizontal_start_y + j, 40, LANE_WIDTH))

    # -------- 黑白格终点线 --------
    finish_line_y = 40
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


if __name__ == "__main__":
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("90° Turn Track")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        draw_track(win)
        pygame.display.update()