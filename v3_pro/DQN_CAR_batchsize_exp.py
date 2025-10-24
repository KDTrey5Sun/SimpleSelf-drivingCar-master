import pygame
import numpy as np
import statistics
from autocar_v3 import ComputerCar, FPS, HEIGHT, CENTER_X
from DQN import Agent
from draw_v3 import draw, WIDTH, HEIGHT, CENTER_X

"""
批大小 (batch_size) 对 DQN 学习效率与稳定性的影响实验脚本。

与 DQN_CAR_v3 copy 1.py (用于比较 replay buffer 容量) 区别：
  - 固定 replay buffer 容量 max_mem_size
  - 变化 batch_size (每次学习采样量)
  - 记录 reward / loss / success_rate / epsilon 曲线
  - 输出统一格式: batch_size,metric,idx,value 到 ./v3/curve_data_batchsize.txt

保持判定成功标准 reward == 3000 (可按环境需要修改)。
"""

def run_batchsize_experiments(
    batch_sizes,
    max_mem_size=50000,
    max_success=100,
    gamma=0.99,
    init_epsilon=0.9,
    eps_end=0.05,
    eps_dec=0.001,
    lr=3e-4,
    learn_starts=1000,
    replace_target=2000,
    success_reward=3000,
    output_path='./v3_pro/curve_data_batchsize.txt'
):
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("DQN Car v3 - Batch Size Experiment")
    clock = pygame.time.Clock()

    results = {}
    all_curve_data = []

    for bs in batch_sizes:
        print(f"\n==== Running experiments with batch_size: {bs} (replay capacity={max_mem_size}) ====")
        scores = []
        tries_list = []
        reward_history = []
        loss_history = []
        success_history = []
        epsilon_history = []
        success_count = 0
        episode_idx = 0
        tries_since_last_success = 0

        agent = Agent(
            gamma=gamma,
            epsilon=init_epsilon,
            batch_size=bs,
            n_actions=4,
            eps_end=eps_end,
            input_dims=4,
            lr=lr,
            max_mem_size=max_mem_size,
            eps_dec=eps_dec,
            combined=False,
            # learn_starts=learn_starts,
            # replace_target=replace_target,
            # double_dqn=True,
            # dueling=True
        )

        while success_count < max_success:
            env = ComputerCar(max_vel=8, rotation_vel=4)
            # env.set_start_pos((CENTER_X, HEIGHT - 80))
            env.reset()

            score = 0
            done = False
            observation = env.reset_env()
            episode_loss = []

            while not done:
                clock.tick(FPS)
                draw(win, env)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                action = agent.choose_action(observation)
                observation_, reward, done = env.step(action)
                score += reward
                agent.memory.store_transition(observation, action, reward, observation_, done)
                observation = observation_

                loss = agent.learn()
                if loss is not None:
                    episode_loss.append(loss)

            epsilon_history.append(agent.epsilon)
            reward_history.append(score)
            loss_history.append(np.mean(episode_loss) if episode_loss else 0)
            scores.append(score)

            if reward == success_reward:
                success_count += 1
                tries_list.append(tries_since_last_success)
                success_history.append(1)
                print(f'[batch {bs}] success #{success_count} | ep {episode_idx} | score {score:.1f} | tries_since_last {tries_since_last_success} | eps {agent.epsilon:.3f}')
                tries_since_last_success = 0
            else:
                tries_since_last_success += 1
                success_history.append(0)
                print(f'[batch {bs}] fail          | ep {episode_idx} | score {score:.1f} | tries {tries_since_last_success} | eps {agent.epsilon:.3f}')
            episode_idx += 1

        avg_score = sum(scores) / len(scores)
        std_score = statistics.stdev(scores) if len(scores) > 1 else 0
        avg_tries = sum(tries_list) / len(tries_list)
        std_tries = statistics.stdev(tries_list) if len(tries_list) > 1 else 0
        results[bs] = dict(avg=avg_score, std=std_score, avg_tries=avg_tries, std_tries=std_tries)
        print(f"\n[Batch {bs}] Average score: {avg_score:.2f} | Std: {std_score:.2f} | Avg tries: {avg_tries:.2f} | Tries std: {std_tries:.2f}")

        # 曲线数据写入缓存
        for idx, v in enumerate(reward_history):
            all_curve_data.append(f"{bs},reward,{idx},{v}\n")
        for idx, v in enumerate(loss_history):
            all_curve_data.append(f"{bs},loss,{idx},{v}\n")
        for idx, v in enumerate(epsilon_history):
            all_curve_data.append(f"{bs},epsilon,{idx},{v}\n")
        # success_rate (滑动窗口)
        window = 20
        for idx in range(len(success_history)):
            rate = sum(success_history[max(0, idx-window+1):idx+1]) / min(idx+1, window)
            all_curve_data.append(f"{bs},success_rate,{idx},{rate}\n")

    print("\n==== Batch size experiment summary ====")
    for b, m in results.items():
        print(f"Batch {b} | Avg score {m['avg']:.2f} | Std {m['std']:.2f} | Avg tries {m['avg_tries']:.2f} | Tries std {m['std_tries']:.2f}")

    with open(output_path, 'w') as f:
        f.writelines(all_curve_data)
    print(f"Curve data saved to {output_path}")
    pygame.quit()


if __name__ == '__main__':
    # 可根据需要调整批大小列表
    batch_sizes = [4, 64, 128, 1280]
    run_batchsize_experiments(batch_sizes)
