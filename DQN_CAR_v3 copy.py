import pygame
import numpy as np
from autocar_v3 import ComputerCar, FPS
from DQN_v1 import Agent
import statistics
from draw_v3 import draw, draw_track, WIDTH, HEIGHT, TRACK_WIDTH, LANE_WIDTH, CENTER_X

if __name__ == '__main__':
    pygame.init()
    WIN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("DQN Car v3 - 直线赛道")
    clock = pygame.time.Clock()

    # buffer_sizes = [64, 128, 256]
    buffer_sizes = [8, 16, 32]
    n_experiments = 100
    results = {}

    for buffer_size in buffer_sizes:
        print(f"\n==== Running experiments with replay buffer size: {buffer_size} ====")
        scores = []
        tries_list = []

        for i in range(n_experiments):
            
            env = ComputerCar(max_vel=8, rotation_vel=4)
            env.START_POS = (CENTER_X, HEIGHT - 80)
            env.reset()

            # agent = Agent(
            #     gamma=0.95,
            #     epsilon=1,
            #     batch_size=buffer_size,
            #     n_actions=4,
            #     eps_end=0.1,
            #     input_dims=4,
            #     lr=0.0005,
            #     max_mem_size=50000,
            #     eps_dec=0.002,
            #     combined=False
            # )

            agent = Agent(
                gamma=0.99,
                epsilon=0.8,
                batch_size=buffer_size,
                n_actions=4,
                eps_end=0.05,
                input_dims=4,
                lr=0.0003,
                max_mem_size=100000,
                eps_dec=0.01,
                combined=False
            )

            score = 0
            idx = 0
            done = False
            observation = env.reset_env()

            while not done:
                clock.tick(FPS)
                draw(WIN, env)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        exit()

                action = agent.choose_action(observation)
                observation_, reward, done = env.step(action)
                score += reward
                agent.memory.store_transition(observation, action, reward, observation_, done)
                observation = observation_

                if idx % 1 == 0:
                    agent.learn()
                idx += 1

            print(f'buffer_size: {buffer_size}, episode: {i}, score: {score:.2f}, tries: {idx}')
            scores.append(score)
            tries_list.append(idx)

        avg_score = sum(scores) / len(scores)
        std_score = statistics.stdev(scores) if len(scores) > 1 else 0
        avg_tries = sum(tries_list) / len(tries_list)
        std_tries = statistics.stdev(tries_list) if len(tries_list) > 1 else 0
        results[buffer_size] = {
            'avg': avg_score,
            'std': std_score,
            'avg_tries': avg_tries,
            'std_tries': std_tries
        }
        print(f"\n[Buffer size {buffer_size}] 平均得分: {avg_score:.2f}, 一致性(标准差): {std_score:.2f}, 平均试错次数: {avg_tries:.2f}, 试错一致性(标准差): {std_tries:.2f}")

    print("\n==== 所有实验结果 ====")
    for buf, metrics in results.items():
        print(f"Replay buffer: {buf} | 平均得分: {metrics['avg']:.2f} | 一致性(标准差): {metrics['std']:.2f} | 平均试错次数: {metrics['avg_tries']:.2f} | 试错一致性(标准差): {metrics['std_tries']:.2f}")

    pygame.quit()