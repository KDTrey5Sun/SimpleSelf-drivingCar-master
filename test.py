import numpy as np
import pygame
import cv2
from utils import scale_image

# 初始化pygame
pygame.init()

# 加载并缩放赛道边界图像
TRACK_BORDER = scale_image(pygame.image.load("imgs/track_border1.png"), 0.3)

def get_track_border_points(surface):
    arr = pygame.surfarray.array3d(surface)
    arr = np.transpose(arr, (1, 0, 2))
    mask = np.any(arr != 0, axis=2)
    points = np.argwhere(mask)
    return points

# 提取边界点
TRACK_BORDER_POINTS = get_track_border_points(TRACK_BORDER)

print("TRACK_BORDER_POINTS shape:", TRACK_BORDER_POINTS.shape)
print(TRACK_BORDER_POINTS[:10])  # 打印前10个边界点坐标


# 设置窗口尺寸
WINDOW_WIDTH, WINDOW_HEIGHT = 682, 932
WIN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Track Border Points Visualization")

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# 绘制坐标轴和网格
def draw_grid_and_axes(surface, width, height, step=50):
    # 绘制网格
    for x in range(0, width, step):
        pygame.draw.line(surface, (50, 50, 50), (x, 0), (x, height), 1)
    for y in range(0, height, step):
        pygame.draw.line(surface, (50, 50, 50), (0, y), (width, y), 1)
    
    # 绘制坐标轴
    pygame.draw.line(surface, RED, (0, height // 2), (width, height // 2), 2)  # X轴
    pygame.draw.line(surface, GREEN, (width // 2, 0), (width // 2, height), 2)  # Y轴
    
    # 标记坐标原点 (0,0) 在左上角（Pygame坐标系）
    font = pygame.font.SysFont(None, 24)
    origin_text = font.render("(0,0)", True, WHITE)
    surface.blit(origin_text, (10, 10))

# ...existing code...

# 取出一组纵坐标相等的四个点
def get_four_points_with_same_y(points):
    # points: (N, 2), 每行为[y, x]
    unique_ys, counts = np.unique(points[:, 0], return_counts=True)
    for y in unique_ys:
        idx = np.where(points[:, 0] == y)[0]
        if len(idx) >= 4:
            return points[idx[:4]]
    return None

four_points = get_four_points_with_same_y(TRACK_BORDER_POINTS)
print("四个纵坐标相等的点：", four_points)
# ...existing code...
# ...existing code...


# 取出一组横坐标相等的四个点
def get_four_points_with_same_x(points):
    # points: (N, 2), 每行为[y, x]
    unique_xs, counts = np.unique(points[:, 1], return_counts=True)
    for x in unique_xs:
        idx = np.where(points[:, 1] == x)[0]
        if len(idx) >= 4:
            return points[idx[:4]]
    return None

four_points_x = get_four_points_with_same_x(TRACK_BORDER_POINTS)
print("四个横坐标相等的点：", four_points_x)
# ...existing code...


# ...existing code...

def estimate_track_width(points):
    """
    points: (N, 2), 每行为[y, x]
    返回估算的跑道宽度（像素）
    """
    xs = np.unique(points[:, 1])
    widths = []
    for x in xs:
        ys = points[points[:, 1] == x][:, 0]
        if len(ys) > 1:
            width = ys.max() - ys.min()
            widths.append(width)
    if len(widths) == 0:
        return None
    return np.median(widths)  # 也可以用np.mean(widths)

track_width = estimate_track_width(TRACK_BORDER_POINTS)
print("估算的跑道宽度为（像素）：", track_width)
# ...existing code...



# 主循环
def main():
    running = True
    clock = pygame.time.Clock()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # 清屏
        WIN.fill(BLACK)
        
        # 绘制网格和坐标轴
        draw_grid_and_axes(WIN, WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # 绘制所有边界点（用蓝色小点表示）
        for point in TRACK_BORDER_POINTS:
            pygame.draw.circle(WIN, BLUE, (point[1], point[0]), 1)  # 注意：point是(y,x)格式
        
        # 显示当前鼠标位置
        mouse_x, mouse_y = pygame.mouse.get_pos()
        font = pygame.font.SysFont(None, 24)
        mouse_text = font.render(f"Mouse: ({mouse_x}, {mouse_y})", True, WHITE)
        WIN.blit(mouse_text, (WINDOW_WIDTH - 150, 10))
        
        pygame.display.update()
        clock.tick(30)
    
    pygame.quit()

if __name__ == "__main__":
    main()