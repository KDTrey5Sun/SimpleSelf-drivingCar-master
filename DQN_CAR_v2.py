from autocar_v2 import *
from DQN_v1 import Agent
import torch
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt


def draw(win, images, car):
    for img, pos in images:
        win.blit(img, pos)
    car.draw(win)
    pygame.display.update()

if __name__ == '__main__':
    run = True
    clock = pygame.time.Clock()
    images = [(GRASS, (0, 0)), (TRACK, (0, 0)),
              (FINISH, FINISH_POSITION), (TRACK_BORDER, (0, 0))]

    # env = ComputerCar(1, 1)
    env = ComputerCar(max_vel=400, rotation_vel=4)  # max_vel和rotation_vel都适当加大
    
    combined = False
    buffer_size = 50000

    agent = Agent(gamma=0.95, 
                  epsilon=1, 
                  batch_size=128,  # <--- 原来是64
                  n_actions=4,  # <--- 改为4
                  eps_end=0.1, 
                  input_dims=4, 
                  lr=0.0005,
                  max_mem_size=buffer_size, 
                  eps_dec=0.002,
                  combined=combined)

    # agent.Q_eval.load_state_dict(torch.load('weight_eval.pt'))
    # agent.Q_next.load_state_dict(torch.load('weight_next.pt'))

    scores = []
    n_games = 10000


    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'b-')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.set_title('Training Score Curve')

    for i in range(n_games):
        score = 0
        idx = 0
        done = False
        observation = env.reset_env()  # <--- 使用 reset_env()

        while not done:
            clock.tick(FPS)
            draw(WIN, images, env)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

            if not run:
                break

            action = agent.choose_action(observation)
            # print(f'Action chosen: {action}')  # <--- 打印动作

            observation_, reward, done = env.step(action)
            score += reward
            agent.memory.store_transition(observation, action, reward,
                                          observation_, done)
            # print(f'Observation: {observation}, Action: {action}, Reward: {reward}')  # <--- 打印状态、动作和奖励

            observation = observation_

            # if idx % 100 == 0:
            if idx % 1 == 0:
                agent.learn()
            idx += 1
        print("-------------------------------------------------------------")
        print(f'episode: {i}, score: {score:.2f}')
        scores.append(score)
        
        
        # df = pd.DataFrame({'Episode': list(range(1, len(scores)+1)), 'Score': scores})
        # print("\033c", end="")  # 清屏
        # print(tabulate(df.tail(20), headers='keys', tablefmt='psql', showindex=False))  # 只显示最近20轮
        
        line.set_xdata(range(1, len(scores)+1))
        line.set_ydata(scores)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)

        # 保存网络
        if i > 100:
            torch.save(agent.Q_eval.state_dict(), 'weight_eval.pt')
            torch.save(agent.Q_next.state_dict(), 'weight_next.pt')
            # agent.memory.save_buffer('buffer')  # 如 memory 实现支持

        if not run:
            break

    pygame.quit()
