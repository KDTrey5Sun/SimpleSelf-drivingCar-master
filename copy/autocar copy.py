from dis import dis
from telnetlib import PRAGMA_HEARTBEAT
from turtle import Screen
from charset_normalizer import detect
from numpy import size
import numpy as np
import pygame
import time
import math
from utils import scale_image, blit_rotate_center
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
WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Racing Game!")
FPS = 30
PATH = [(501, 314), (498, 221), (462, 142), (403, 93), (325, 73), (247, 96), (184, 152), (149, 237), (148, 353), (150, 459), (148, 565),
        (148, 672), (180, 771), (238, 828), (324, 853), (409, 832), (470, 776), (501, 678), (503, 568), (500, 516), (499, 467)]
class AbstractCar:
    def __init__(self, max_vel, rotation_vel):
        self.img = self.IMG
        self.max_vel = max_vel
        self.vel = 1
        self.rotation_vel = rotation_vel
        self.angle = 0
        self.x, self.y = self.START_POS
        self.acceleration = 0.1
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


class ComputerCar(AbstractCar):
    IMG = GREEN_CAR
    START_POS = (488, 370)

    def __init__(self, max_vel, rotation_vel, path=[]):
        super().__init__(max_vel, rotation_vel)
        self.path = path
        self.current_point = 0
        self.car_rect = (0, 0, 0, 0)
        self.N_rect = (0, 0, 0, 0)
        self.S_rect = (0, 0, 0, 0)
        self.E_rect = (0, 0, 0, 0)
        self.W_rect = (0, 0, 0, 0)
        self.to_target = False            
        self.is_finished = False          
        self.is_collide = False         
        self.cumulated_rewards = 0       
        self.__calculate_x_y()            

        self.idx = 0
        self.dist_ls = [[0,0],[0,0]]      
        self.direc = 0                    
    def __calculate_x_y(self):
        self.cx = self.x + self.img.get_width() / 2
        self.cy = self.y + self.img.get_height() / 2

    def __calculate_dist(self):
        target_x, target_y = self.path[self.current_point]
        x_diff = target_x - self.cx
        y_diff = target_y - self.cy
        if self.idx >= 5:
            self.dist_ls[0][0] = self.dist_ls[1][0]
            self.dist_ls[0][1] = self.dist_ls[1][1]
            self.dist_ls[1][0] = x_diff
            self.dist_ls[1][1] = y_diff
            self.idx = 0
        self.idx += 1
        A_x, A_y = self.path[self.current_point - 1]
        B_x, B_y = self.path[self.current_point]
        C_x = self.cx
        C_y = self.cy
        dc_x = B_x - A_x
        dc_y = B_y - A_y

        db_x = C_x - A_x
        db_y = C_y - A_y

        cross_prdct = db_x*dc_y - db_y*dc_x
        if cross_prdct > 0:
            direction = 1
        elif cross_prdct == 0:
            direction = 0
        else:
            direction = -1
        self.direc = direction
        c = math.sqrt(dc_x**2 + dc_y**2) 
        b = math.sqrt(db_x**2 + db_y**2)  
        theta = math.acos((dc_x*db_x + dc_y*db_y)/(b*c))  
        dev = abs(b*math.sin(theta)) 
        if dc_y != 0:
            phi = math.atan(dc_x/dc_y)*180/math.pi
            if dc_y < 0:
                phi = phi
            elif dc_x < 0:
                phi = 180 - abs(phi)
            else:
                phi = -180 + abs(phi)
        else:
            if dc_x < 0:
                phi = 90
            else:
                phi = -90


        beta = self.angle - phi
        if (beta > 180) or (beta < -180):
            if phi >= 0:
                beta = 360 - abs(beta)
            else:
                beta = abs(beta) - 360
        return beta, dev, direction

    def __get_rewards(self):
        car_mask1 = pygame.mask.from_surface(self.img)
        offset = (int(self.x - 0), int(self.y - 0))
        poi = TRACK_BORDER_MASK.overlap(car_mask1, offset)

        if poi != None: 
            is_collided = 1
            self.is_collide = True
        else: 
            is_collided = 0
            self.is_collide = False


        beta, dev, direction = self.__calculate_dist()

        dist_0 = math.sqrt(self.dist_ls[0][0] ** 2 + self.dist_ls[0][1] ** 2)
        dist_1 = math.sqrt(self.dist_ls[1][0] ** 2 + self.dist_ls[1][1] ** 2)

        ddist = dist_1 - dist_0 
        if dev < 2:
            rewards = dev * 0.3 - 0.05 * abs(beta) - 1 / (ddist * 10 + 200)
        else:
            rewards = -dev * 0.3 - 0.05 * abs(beta) - 1 / (ddist * 10 + 200)


        if is_collided:
            rewards = rewards - 2000

        if self.to_target:
            if self.current_point > 7:
                rewards = 1300
            else:
                rewards = 800

        if self.is_finished:
            rewards = 100000
        return rewards, beta, dev, direction

    def step(self, keys):
        moved = False

        if keys == 0:
            self.rotate(left=True)
            moved = True
            self.move_forward()
        if keys == 1:
            self.rotate(right=True)
            moved = True
            self.move_forward()

        if not moved:
            self.reduce_speed()

        self.__handle_collision()

        reward, beta, dev, direction= self.__get_rewards()
        self.cumulated_rewards += reward

        if self.cumulated_rewards < -1000:
            done = True
        elif self.is_finished:
            done = True
        elif self.is_collide:
            done = True
        else:
            done = False
        return ([beta, dev, direction], reward, done)

    def __handle_collision(self):
        if self.collide(TRACK_BORDER_MASK) != None:
            self.bounce()

        player_finish_poi_collide = self.collide(
            FINISH_MASK, *FINISH_POSITION)
        if player_finish_poi_collide != None:
            if player_finish_poi_collide[1] == 0:
                self.bounce()
            else:
                self.reset()
                self.is_finished = True

    def reduce_speed(self):
        self.vel = max(self.vel - self.acceleration / 2, 0)
        self.move()

    def draw_points(self, win):
        for point in self.path:
            pygame.draw.circle(win, (255, 0, 0), point, 5)
        self.__calculate_x_y()
        pygame.draw.circle(win, (255, 0, 0), (self.cx, self.cy), 2)

    def draw_car_rect(self, win):
        pygame.draw.rect(win, (0, 255, 0), self.car_rect, 1)
        pygame.draw.rect(win, (0, 0, 255), self.N_rect, 1)
        pygame.draw.rect(win, (0, 0, 255), self.S_rect, 1)
        pygame.draw.rect(win, (0, 0, 255), self.E_rect, 1)
        pygame.draw.rect(win, (0, 0, 255), self.W_rect, 1)

    def draw(self, win):
        super().draw(win)
        self.draw_points(win)
        self.draw_car_rect(win)

    def bounce(self):
        self.vel = -self.vel * 0.7
        self.move()

    def update_path_point(self):
        to_target = False
        if self.current_point >= len(self.path) - 1:
            self.current_point = 0
            return
        target = self.path[self.current_point]
        rect = pygame.Rect(
            self.x - 15, self.y - 10, self.img.get_width() + 30, self.img.get_height() + 20)

        self.car_rect = ((rect[0], rect[1]), (rect[2], rect[3]))
        # print(self.car_rect,self.current_point)
        if rect.collidepoint(*target):
            self.current_point += 1
            to_target = True
        self.to_target = to_target

    def move(self):
        self.update_path_point()
        super().move()

    def reset(self):
        self.is_finished = False
        self.current_point = 0
        self.cumulated_rewards = 0
        super().reset()
        return np.array([0, 0, 0])
    


    def step(self):
        self.__calculate_x_y()
        vision = self.__sense_environment()

        if vision["front_clear"]:
            self.move_forward()
        elif vision["left_clear"]:
            self.rotate(left=True)
            self.move_forward()
        elif vision["right_clear"]:
            self.rotate(right=True)
            self.move_forward()
        else:
            # 四面碰壁，随机选择一个方向转弯
            self.rotate(left=np.random.rand() > 0.5)
            self.move_forward()

        self.__handle_collision()
        reward, _, _, _ = self.__get_rewards()
        self.cumulated_rewards += reward

        done = self.cumulated_rewards < -1000 or self.is_finished or self.is_collide
        return ([], reward, done)

    def __sense_environment(self):
        """简单模拟3个方向的‘视野’检测：前、左、右"""
        sensor_length = 15
        angle_offsets = [0, -45, 45]  # 前、左、右（单位：角度）
        directions = ["front_clear", "left_clear", "right_clear"]
        status = {}

        for angle_offset, dir_name in zip(angle_offsets, directions):
            sensor_angle = math.radians(self.angle + angle_offset)
            dx = math.sin(sensor_angle) * sensor_length
            dy = math.cos(sensor_angle) * sensor_length
            sense_x = int(self.x - dx)
            sense_y = int(self.y - dy)

            if 0 <= sense_x < TRACK_BORDER_MASK.get_size()[0] and 0 <= sense_y < TRACK_BORDER_MASK.get_size()[1]:
                collision = TRACK_BORDER_MASK.get_at((sense_x, sense_y))
                status[dir_name] = collision == 0  # 0 表示没有碰撞
            else:
                status[dir_name] = False  # 出界视为有障碍
        return status



def draw(win, images, player_car):
    for img, pos in images:
        win.blit(img, pos)

    player_car.draw(win)
    pygame.display.update()
