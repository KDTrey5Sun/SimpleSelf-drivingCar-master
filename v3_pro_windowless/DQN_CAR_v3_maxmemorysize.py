import os
import sys
import time
import pygame
import numpy as np
import statistics
from autocar_v3 import ComputerCar, FPS
from DQN import Agent
from draw_v3 import WIDTH, HEIGHT, CENTER_X

def run_buffersize_experiments(
    buffer_sizes,
    batch_size=128,
    max_success=50,
    gamma=0.99,
    init_epsilon=0.9,
    eps_end=0.05,
    eps_dec=0.001,
    lr=3e-4,
    learn_starts=2000,
    replace_target=2000,
    success_reward=3000.0,
    output_curve='./v3_pro_windowless/memory_size_data/curve_data_buffersize.txt',
    output_summary='./v3_pro_windowless/memory_size_data/summary_buffersize.txt',
    output_log='./v3_pro_windowless/memory_size_data/train_log_buffersize.txt',
    windowless=True,
    max_episodes=None,
    print_every=1):



    if windowless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    # 确保输出目录存在
    for p in [output_curve, output_summary, output_log]:
        d = os.path.dirname(os.path.abspath(p))
        os.makedirs(d, exist_ok=True)

    pygame.init()
    win = pygame.display.set_mode((max(1, WIDTH), max(1, HEIGHT)))
    pygame.display.set_caption("DQN Car v3_pro_windowless - Buffer Size Experiment")
    clock = pygame.time.Clock()

    def log_print(fh, msg):
        print(msg)
        fh.write(msg + "\n")
        fh.flush()

    all_curve_data = []
    summary_lines = []
    results = {}

    with open(output_log, 'w') as log_f:
        for cap in buffer_sizes:
            log_print(log_f, f"\n==== Start experiment: replay capacity (max_mem_size)={cap}, batch_size={batch_size} ====")

            scores = []
            tries_list = []
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
                batch_size=batch_size,
                n_actions=4,
                eps_end=eps_end,
                input_dims=4,
                lr=lr,
                max_mem_size=cap,
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
                if (max_episodes is not None) and (episode_idx >= max_episodes):
                    log_print(log_f, f"[buffer {cap}] reach max_episodes={max_episodes}, stop this capacity.")
                    break

                env = ComputerCar(max_vel=8, rotation_vel=4)
                # env.START_POS = (CENTER_X, HEIGHT - 80)
                env.reset()

                score = 0.0
                done = False
                observation = env.reset_env()
                episode_loss = []
                step_count = 0
                ep_start = time.time()
                last_reward = 0.0

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

                eps_value = agent.epsilon
                loss_mean = float(np.mean(episode_loss) if episode_loss else 0.0)
                epsilon_history.append(eps_value)
                reward_history.append(score)
                loss_history.append(loss_mean)
                scores.append(score)

                mem_cntr = getattr(agent.memory, 'mem_cntr', None)
                mem_size = getattr(agent.memory, 'mem_size', None)
                mem_info = f"{mem_cntr}/{mem_size}" if (mem_cntr is not None and mem_size is not None) else str(mem_cntr)

                took = time.time() - ep_start

                # 成功判断：到达终点标志，或最后一步 reward == 终点奖励
                succeeded = getattr(env, 'is_finished', False) or (last_reward >= success_reward - 1e-6)

                if succeeded:
                    prev_tries = tries_since_last_success
                    success_id = success_count + 1
                    success_count += 1
                    tries_list.append(prev_tries)
                    success_history.append(1)

                    log_print(log_f,
                        f"[buffer {cap}] >>> SUCCESS #{success_id} | ep {episode_idx} | "
                        f"score {score:.2f} | last_reward {last_reward:.1f} | loss {loss_mean:.3f} | eps {eps_value:.3f} | "
                        f"steps {step_count} | mem {mem_info} | tries_since_last={prev_tries} | {took*1000:.0f} ms"
                    )
                    status = "SUCCESS"
                    tries_note = f"tries_since_last={prev_tries}"
                    tries_since_last_success = 0
                else:
                    tries_since_last_success += 1
                    success_history.append(0)
                    status = "fail"
                    tries_note = f"tries={tries_since_last_success}"

                if (episode_idx % print_every) == 0:
                    log_print(log_f,
                        f"[buffer {cap}] ep {episode_idx:5d} | {status:7s} | "
                        f"score {score:9.2f} | last_reward {last_reward:6.1f} | loss {loss_mean:7.3f} | eps {eps_value:5.3f} | "
                        f"steps {step_count:4d} | mem {mem_info} | {tries_note} | {took*1000:.0f} ms"
                    )

                episode_idx += 1
            
            episodes = len(scores)
            avg_score = sum(scores) / episodes if episodes else 0.0
            std_score = statistics.stdev(scores) if episodes > 1 else 0.0
            avg_tries = sum(tries_list) / len(tries_list) if tries_list else 0.0
            std_tries = statistics.stdev(tries_list) if len(tries_list) > 1 else 0.0
            took_total = time.time() - t0
            succ_rate = (success_count / episodes) if episodes > 0 else 0.0
            epm = episodes / (took_total / 60.0) if took_total > 0 else 0.0
            avg_loss = float(np.mean(loss_history)) if loss_history else 0.0
            avg_eps = float(np.mean(epsilon_history)) if epsilon_history else agent.epsilon

            results[cap] = dict(
                avg=avg_score, std=std_score, avg_tries=avg_tries, std_tries=std_tries,
                episodes=episodes, successes=success_count, seconds=took_total,
                succ_rate=succ_rate, epm=epm, avg_loss=avg_loss, avg_eps=avg_eps
            )

            # Summary
            per_buf_summary = (
                f"==== Summary (capacity={cap}) ====\n"
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
                f"\n"
            )
            summary_lines.append(per_buf_summary)
            log_print(log_f, "\n" + per_buf_summary)

            # 曲线数据
            for idx, v in enumerate(reward_history):
                all_curve_data.append(f"{cap},reward,{idx},{v}\n")
            for idx, v in enumerate(loss_history):
                all_curve_data.append(f"{cap},loss,{idx},{v}\n")
            for idx, v in enumerate(epsilon_history):
                all_curve_data.append(f"{cap},epsilon,{idx},{v}\n")
            window = 20
            for idx in range(len(success_history)):
                rate = sum(success_history[max(0, idx - window + 1): idx + 1]) / min(idx + 1, window)
                all_curve_data.append(f"{cap},success_rate,{idx},{rate}\n")

    # 写曲线与分容量摘要
    with open(output_curve, 'w') as f:
        f.writelines(all_curve_data)
    with open(output_summary, 'w') as f:
        f.writelines(summary_lines)

    # 最终汇总：逐个 size 的 Summary 块（与单个 buffer 的格式一致）
    final_lines = []
    final_lines.append("==== Final Summary (all buffer sizes) ====\n\n")
    for cap in sorted(results.keys()):
        m = results[cap]
        block = (
            f"==== Summary (capacity={cap}) ====\n"
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
            f"\n"
        )
        final_lines.append(block)

    with open(output_summary, 'a') as f:
        f.writelines(final_lines)

    print("".join(final_lines))
    print(f"Curve data saved to {output_curve}")
    print(f"Summary saved to {output_summary}")
    pygame.quit()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        buffer_sizes = [int(x) for x in sys.argv[1:]]
    else:
        buffer_sizes = [1000, 5000, 20000, 100000]
    run_buffersize_experiments(
        buffer_sizes=buffer_sizes,
        batch_size=128,
        windowless=True,
        print_every=1,
        max_success=50,
        # max_episodes=40000,  # 可选：限制最大训练局数
    )