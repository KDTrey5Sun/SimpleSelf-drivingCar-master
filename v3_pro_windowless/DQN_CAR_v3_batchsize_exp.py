import os
import time
import pygame
import numpy as np
import statistics
from autocar_v3 import ComputerCar, FPS
from DQN import Agent
from draw_v3 import WIDTH, HEIGHT, CENTER_X

def run_batchsize_experiments(
    batch_sizes,
    fixed_mem_size=100000,
    max_success=100,
    gamma=0.99,
    init_epsilon=0.9,
    eps_end=0.05,
    eps_dec=0.001,
    lr=3e-4,
    learn_starts=2000,
    replace_target=2000,
    success_reward=3000.0,
    output_curve='./v3_pro_windowless/batch_size_data/curve_data_batchsize.txt',
    output_summary='./v3_pro_windowless/batch_size_data/summary_batchsize.txt',
    output_log='./v3_pro_windowless/batch_size_data/train_log_batchsize.txt',
    windowless=True,
):
    if windowless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    # 确保输出目录存在
    for p in [output_curve, output_summary, output_log]:
        d = os.path.dirname(os.path.abspath(p))
        os.makedirs(d, exist_ok=True)

    pygame.init()
    win = pygame.display.set_mode((max(1, WIDTH), max(1, HEIGHT)))
    pygame.display.set_caption("DQN Car v3_pro_windowless - Batch Size Experiment")
    clock = pygame.time.Clock()

    results = {}
    all_curve_data = []

    with open(output_log, 'w') as log_f, open(output_summary, 'w') as summary_f:
        for bs in batch_sizes:
            print(f"\n==== Running experiments with batch_size: {bs} (fixed replay capacity={fixed_mem_size}) ====")
            log_f.write(f"\n==== Running experiments with batch_size: {bs} (fixed replay capacity={fixed_mem_size}) ====\n")
            scores = []
            tries_list = []
            attempts_list = []
            reward_history = []
            loss_history = []
            success_history = []
            epsilon_history = []
            success_count = 0
            episode_idx = 0
            tries_since_last_success = 0
            t0 = time.time()

            agent = Agent(
                gamma=gamma,
                epsilon=init_epsilon,
                batch_size=bs,
                n_actions=4,
                eps_end=eps_end,
                input_dims=4,
                lr=lr,
                max_mem_size=fixed_mem_size,
                eps_dec=eps_dec,
                combined=False,
                # learn_starts=learn_starts,
                # replace_target=replace_target,
                # double_dqn=True,
                # dueling=True,
                # clip_reward=False,
                # target_soft_tau=0.0
            )

            while success_count < max_success:
                env = ComputerCar(max_vel=8, rotation_vel=4)
                # env.START_POS = (CENTER_X, HEIGHT - 80)
                env.reset()

                score = 0.0
                done = False
                observation = env.reset_env()
                episode_loss = []
                last_reward = 0.0
                step_count = 0

                while not done:
                    if not windowless:
                        clock.tick(FPS)
                    else:
                        pygame.event.pump()

                    action = agent.choose_action(observation)
                    observation_, reward, done = env.step(action)
                    last_reward = reward
                    score += reward
                    agent.memory.store_transition(observation, action, reward, observation_, done)
                    observation = observation_

                    loss = agent.learn()
                    if loss is not None:
                        episode_loss.append(loss)
                    step_count += 1

                epsilon_history.append(agent.epsilon)
                reward_history.append(score)
                loss_history.append(float(np.mean(episode_loss) if episode_loss else 0.0))
                scores.append(score)

                succeeded = getattr(env, 'is_finished', False) or (last_reward >= success_reward - 1e-6)

                if succeeded:
                    attempts = tries_since_last_success + 1
                    success_count += 1
                    tries_list.append(attempts)
                    attempts_list.append(attempts)
                    success_history.append(1)
                    log_line = (
                        f"[batch {bs}] >>> SUCCESS #{success_count} | ep {episode_idx} | "
                        f"score {score:.1f} | last_reward {last_reward:.1f} | loss {loss_history[-1]:.3f} | "
                        f"eps {agent.epsilon:.3f} | steps {step_count} | attempts {attempts} (fails {tries_since_last_success}+1)\n"
                    )
                    print(log_line.strip())
                    log_f.write(log_line)
                    tries_since_last_success = 0
                else:
                    tries_since_last_success += 1
                    success_history.append(0)
                    log_line = (
                        f"[batch {bs}] fail          | ep {episode_idx} | "
                        f"score {score:.1f} | last_reward {last_reward:.1f} | loss {loss_history[-1]:.3f} | "
                        f"eps {agent.epsilon:.3f} | steps {step_count} | fails_since_last {tries_since_last_success}\n"
                    )
                    print(log_line.strip())
                    log_f.write(log_line)
                episode_idx += 1

            episodes = len(scores)
            avg_score = sum(scores) / episodes if episodes else 0.0
            std_score = statistics.stdev(scores) if episodes > 1 else 0.0
            avg_tries = sum(tries_list) / len(tries_list) if tries_list else 0.0
            std_tries = statistics.stdev(tries_list) if len(tries_list) > 1 else 0.0
            succ_rate = (success_count / episodes) if episodes > 0 else 0.0
            avg_loss = float(np.mean(loss_history)) if loss_history else 0.0
            avg_eps = float(np.mean(epsilon_history)) if epsilon_history else agent.epsilon
            took_total = time.time() - t0
            epm = episodes / (took_total / 60.0) if took_total > 0 else 0.0

            results[bs] = dict(
                avg=avg_score, std=std_score, avg_tries=avg_tries, std_tries=std_tries,
                episodes=episodes, successes=success_count, succ_rate=succ_rate,
                avg_loss=avg_loss, avg_eps=avg_eps, seconds=took_total, epm=epm,
            )

            summary_block = (
                f"\n==== Summary (batch={bs}) ====\n"
                f"episodes: {episodes}\n"
                f"successes: {success_count}\n"
                f"success_rate: {succ_rate:.3f}\n"
                f"avg_score: {avg_score:.2f}\n"
                f"std_score: {std_score:.2f}\n"
                f"avg_loss: {avg_loss:.3f}\n"
                f"avg_epsilon: {avg_eps:.3f}\n"
                f"avg_tries: {avg_tries:.2f}\n"
                f"tries_std: {std_tries:.2f}\n"
                f"time_min: {took_total/60:.2f}\n"
                f"episodes_per_min: {epm:.1f}\n"
            )
            print(summary_block)
            summary_f.write(summary_block)

            for idx, v in enumerate(reward_history):
                all_curve_data.append(f"{bs},reward,{idx},{v}\n")
            for idx, v in enumerate(loss_history):
                all_curve_data.append(f"{bs},loss,{idx},{v}\n")
            for idx, v in enumerate(epsilon_history):
                all_curve_data.append(f"{bs},epsilon,{idx},{v}\n")
            window = 20
            for idx in range(len(success_history)):
                rate = sum(success_history[max(0, idx - window + 1): idx + 1]) / min(idx + 1, window)
                all_curve_data.append(f"{bs},success_rate,{idx},{rate}\n")

        # 最终汇总
        summary_f.write("\n==== Final Summary (all batch sizes) ====\n")
        print("\n==== Final Summary (all batch sizes) ====\n")
        for b in sorted(results.keys()):
            m = results[b]
            block = (
                f"==== Summary (batch={b}) ====\n"
                f"episodes: {m['episodes']}\n"
                f"successes: {m['successes']}\n"
                f"success_rate: {m['succ_rate']:.3f}\n"
                f"avg_score: {m['avg']:.2f}\n"
                f"std_score: {m['std']:.2f}\n"
                f"avg_loss: {m['avg_loss']:.3f}\n"
                f"avg_epsilon: {m['avg_eps']:.3f}\n"
                f"avg_tries: {m['avg_tries']:.2f}\n"
                f"tries_std: {m['std_tries']:.2f}\n"
                f"time_min: {m['seconds']/60:.2f}\n"
                f"episodes_per_min: {m['epm']:.1f}\n"
            )
            print(block)
            summary_f.write(block)

    with open(output_curve, 'w') as f:
        f.writelines(all_curve_data)
    print(f"Curve data saved to {output_curve}")
    pygame.quit()

if __name__ == '__main__':
    batch_sizes = [4, 64, 128, 1280]
    run_batchsize_experiments(batch_sizes)