import pygame
import numpy as np
from autocar_v3 import ComputerCar, FPS
from DQN_v1 import Agent
import statistics
from draw_v3 import draw, draw_track, WIDTH, HEIGHT, TRACK_WIDTH, LANE_WIDTH, CENTER_X

if __name__ == '__main__':
    pygame.init()
    WIN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("DQN Car v3 - Straight Track")
    clock = pygame.time.Clock()

    buffer_sizes = [1, 4, 8]
    max_success = 100
    results = {}

    # 用于保存所有曲线数据
    all_curve_data = []

    for buffer_size in buffer_sizes:
        print(f"\n==== Running experiments with replay buffer size: {buffer_size} ====")
        scores = []
        tries_list = []
        reward_history = []
        loss_history = []
        success_history = []
        success_count = 0
        i = 0
        tries_since_last_success = 0


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
        
        while success_count < max_success:
            env = ComputerCar(max_vel=8, rotation_vel=4)
            env.START_POS = (CENTER_X, HEIGHT - 80)
            env.reset()

            

            score = 0
            idx = 0
            done = False
            observation = env.reset_env()
            episode_loss = []

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

                loss = agent.learn()

                if loss is not None:
                    episode_loss.append(loss)
                idx += 1

            reward_history.append(score)
            if episode_loss:
                loss_history.append(np.mean(episode_loss))
            else:
                loss_history.append(0)


            scores.append(score)
            # Only count success when reward == 3000
            if reward == 3000:
                success_count += 1
                tries_list.append(tries_since_last_success)
                success_history.append(1)
                print(f'buffer_size: {buffer_size}, success: {success_count}, episode: {i}, score: {score:.2f}, tries: {tries_since_last_success}')
                tries_since_last_success = 0  # Reset fail counter
            else:
                tries_since_last_success += 1
                success_history.append(0)
                print(f'buffer_size: {buffer_size}, fail, episode: {i}, score: {score:.2f}, tries: {tries_since_last_success}')
            i += 1

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
        print(f"\n[Buffer size {buffer_size}] Average score: {avg_score:.2f}, Std: {std_score:.2f}, Average tries: {avg_tries:.2f}, Tries std: {std_tries:.2f}")

        # 记录reward曲线
        for idx, val in enumerate(reward_history):
            all_curve_data.append(f"{buffer_size},reward,{idx},{val}\n")
        # 记录loss曲线
        for idx, val in enumerate(loss_history):
            all_curve_data.append(f"{buffer_size},loss,{idx},{val}\n")
        # 计算并记录success rate曲线
        window = 20
        success_rate_curve = []
        for idx in range(len(success_history)):
            rate = sum(success_history[max(0, idx-window+1):idx+1]) / min(idx+1, window)
            success_rate_curve.append(rate)
            all_curve_data.append(f"{buffer_size},success_rate,{idx},{rate}\n")

    print("\n==== All experiment results ====")
    for buf, metrics in results.items():
        print(f"Replay buffer: {buf} | Average score: {metrics['avg']:.2f} | Std: {metrics['std']:.2f} | Average tries: {metrics['avg_tries']:.2f} | Tries std: {metrics['std_tries']:.2f}")

    # 写入文本文件
    with open('curve_data.txt', 'w') as f:
        f.writelines(all_curve_data)

    print("Curve data has been saved to curve_data.txt.")

    pygame.quit()