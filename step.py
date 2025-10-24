def step_v2(self, action):
        # 保存前一位置
        self.prev_x, self.prev_y = self.x, self.y

        # 执行动作
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

        # ----- 碰撞检测 -----
        if self.collide(TRACK_BORDER_MASK):
            reward = -1000
            done = True
            return self.get_state(), reward, done

        if self.collide(FINISH_MASK, *FINISH_POSITION):
            reward = 10000
            done = True
            return self.get_state(), reward, done

        # ----- 奖励设计 -----
        distance_delta = self.get_distance_delta()

        if distance_delta > 0.01:
            reward += 5 * distance_delta  # 鼓励移动
        else:
            reward -= 5  # 原地不动惩罚

        # 加速奖励
        reward += 0.2 * self.vel

        return self.get_state(), reward, done


def step_v1(self, action):
    # 保存前一位置
    self.prev_x, self.prev_y = self.x, self.y

    # 执行动作
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

    # ----- 碰撞检测 -----
    if self.collide(TRACK_BORDER_MASK):
        reward = -1000
        done = True
        return self.get_state(), reward, done

    if self.collide(FINISH_MASK, *FINISH_POSITION):
        reward = 10000
        done = True
        return self.get_state(), reward, done

    # ----- 奖励设计 -----
    distance_delta = self.get_distance_delta()
    center_dist = self.get_distance_to_center()
    if distance_delta > 0.1:
        reward += 10 * distance_delta
    else:
        reward -= 10  # 原地惩罚

    reward += 1 * self.vel  # 鼓励加速
    reward -= 2 * center_dist  # 偏离中心线惩罚

    return self.get_state(), reward, done

def step_v2(self, action):   # 有说法但效率过慢，不稳定
    # 保存前一位置
    self.prev_x, self.prev_y = self.x, self.y

    # 执行动作
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

    # ----- 碰撞检测 -----
    if self.collide(TRACK_BORDER_MASK):
        reward = -500  # 撞墙惩罚
        done = True
        return self.get_state(), reward, done

    if self.collide(FINISH_MASK, *FINISH_POSITION):
        reward = 1000  # 终点奖励
        done = True
        print("FINISH!")
        print("reward:", reward)
        return 
    

    # ----- 奖励设计 -----
    distance_delta = self.get_distance_delta()
    center_dist = self.get_distance_to_center()
    # 用TRACK_WIDTH归一化，距离中心线越近奖励越高
    center_reward = 1.5 * (1 - min(center_dist / TRACK_WIDTH, 1.0))

    reward += 5  # 活着奖励
    reward += 0.5 * self.vel  # 速度奖励
    reward += center_reward   # 靠近中心线奖励

    if distance_delta > 0.1:
        reward += 2 * distance_delta  # 鼓励移动
    else:
        reward -= 20  # 原地不动惩罚

    return self.get_state(), reward, done


def get_state_v1(self):
    dx = FINISH_POSITION[0] - self.x
    dy = FINISH_POSITION[1] - self.y
    distance = math.sqrt(dx ** 2 + dy ** 2)
    angle_diff = math.atan2(dy, dx) * 180 / math.pi - self.angle
    angle_diff = (angle_diff + 180) % 360 - 180  # 限制在[-180,180]
    return np.array([angle_diff / 180, self.vel / self.max_vel, distance / 800], dtype=np.float32)

def get_distance_to_border_v1(self):
    """
    计算当前位置到赛道边界的最短距离（像素单位），高效向量化实现
    """
    current_pos = np.array([self.y, self.x])  # 注意TRACK_BORDER_POINTS是[y, x]
    distances = np.linalg.norm(TRACK_BORDER_POINTS - current_pos, axis=1)
    return np.min(distances)