import pygame
import numpy as np
from autocar_v3 import ComputerCar, FPS
from DQN import Agent
import torch
import matplotlib.pyplot as plt
from draw_v3 import draw, draw_track, WIDTH, HEIGHT, TRACK_WIDTH, LANE_WIDTH, CENTER_X


if __name__ == '__main__':
    pygame.init()
    WIN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("DQN Car v3 - 直线赛道")
    clock = pygame.time.Clock()

    # 你可以根据需要调整参数
    env = ComputerCar(max_vel=8, rotation_vel=4)
    env.reset()

    agent = Agent(
        gamma=0.95,
        epsilon=1,
        batch_size=128,
        n_actions=4,
        eps_end=0.1,
        input_dims=4,
        lr=0.0005,
        max_mem_size=50000,
        eps_dec=0.002,
        combined=False
    )

    scores = []
    n_games = 10000

    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'b-')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.set_title('Training Score Curve')

    run = True
    for i in range(n_games):
        score = 0
        idx = 0
        done = False
        observation = env.reset_env()

        while not done:
            clock.tick(FPS)
            draw(WIN, env)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break
            if not run:
                break

            action = agent.choose_action(observation)
            observation_, reward, done = env.step(action)
            score += reward
            agent.memory.store_transition(observation, action, reward, observation_, done)
            observation = observation_

            if idx % 1 == 0:
                agent.learn()
            idx += 1

        print("-------------------------------------------------------------")
        print(f'episode: {i}, score: {score:.2f}')
        scores.append(score)

        line.set_xdata(range(1, len(scores)+1))
        line.set_ydata(scores)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)

        if i > 100:
            torch.save(agent.Q_eval.state_dict(), 'weight_eval.pt')
            torch.save(agent.Q_next.state_dict(), 'weight_next.pt')

        if not run:
            break

    pygame.quit()