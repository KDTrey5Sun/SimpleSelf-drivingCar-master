import os
import time
import random
import pygame
import numpy as np
import statistics
import math

# 固定为包内相对导入，避免被仓库根目录同名文件遮蔽导致类型不确定
from autocar_v4 import ComputerCar, FPS
from DQN import Agent
from draw_v4 import draw_track, TURN_RADIUS, WIDTH, HEIGHT, TRACK_WIDTH, LANE_WIDTH, CENTER_X

# 可调的起点水平偏移量（>0 向右，<0 向左）
START_SHIFT_X = 200

def run_v4_training(
    max_success=100,
    batch_size=128,
    max_mem_size=100000,
    gamma=0.99,
    lr=5e-4,

    # init_epsilon=1.0,
    # eps_end=0.05,
    # eps_dec=1e-4,

    init_epsilon = 1.0,
    eps_end = 0.05,
    # eps_dec = 5e-6,   # 约在 500,000 steps 衰减到底

    eps_dec = 1e-4,  # 约在 10,000 steps 衰减到底

    learn_starts=5000,
    replace_target=2000,
    success_reward=3000.0,
    windowless=True,
    render=False,                  # 设置 True 可可视化训练
    output_curve='./v4/v4_exp_data/curve_data_v4.txt',
    output_summary='./v4/v4_exp_data/summary_v4.txt',
    output_log='./v4/v4_exp_data/train_log_v4.txt',
    print_every=1,
    max_episodes=None,            # 可选兜底
):
    # 头less模式屏蔽窗口/音频
    if windowless and not render:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    # 随机种子（保证可复现）
    seed = int(os.environ.get("SEED", "123"))
    random.seed(seed)
    np.random.seed(seed)

    # 输出目录
    for p in [output_curve, output_summary, output_log]:
        d = os.path.dirname(os.path.abspath(p))
        os.makedirs(d, exist_ok=True)

    pygame.init()
    win = pygame.display.set_mode((max(1, WIDTH), max(1, HEIGHT))) if render else pygame.display.set_mode((1, 1))
    if render:
        pygame.display.set_caption("DQN Car v4 - 新赛道训练")
        track_surface = win.copy()
        draw_track(win)
        track_surface = win.copy()
    clock = pygame.time.Clock()

    def log_print(fh, msg):
        print(msg)
        fh.write(msg + "\n")
        fh.flush()

    # 代理
    agent = Agent(
        gamma=gamma,
        epsilon=init_epsilon,
        batch_size=batch_size,
        n_actions=4,
        eps_end=eps_end,
        input_dims=4,
        lr=lr,
        max_mem_size=max_mem_size,
        eps_dec=eps_dec,
        combined=False,
        learn_starts=learn_starts,
        replace_target=replace_target,
        double_dqn=False,
        dueling=False,
        clip_reward=False,
        target_soft_tau=0.0
        )

    # 统计
    scores = []
    loss_history = []
    epsilon_history = []
    success_history = []
    all_curve_data = []
    success_count = 0
    tries_since_last_success = 0
    attempts_list = []
    episode_idx = 0
    t0 = time.time()

    # checkpoint 相关
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(output_log)), 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    last_ckpt_time = time.time()
    ckpt_interval_sec = 600  # 每10分钟一次

    with open(output_log, 'w') as log_f:
        log_print(log_f, f"==== Start v4 training | batch_size={batch_size}, replay={max_mem_size} ====")

        while success_count < max_success:
            if max_episodes is not None and episode_idx >= max_episodes:
                log_print(log_f, f"reach max_episodes={max_episodes}, stop.")
                break

            # 环境与起点（沿用 v4 的起点计算）
            env = ComputerCar(max_vel=8, rotation_vel=4)
            base_start_x = CENTER_X + 400
            base_start_y = HEIGHT // 2 + TURN_RADIUS - TRACK_WIDTH // 2
            env.START_POS = (base_start_x + START_SHIFT_X, base_start_y)
            # 终点为黑白格子的中心点
            finish_line_y = 40
            finish_line_height = 20
            finish_cx = CENTER_X
            finish_cy = finish_line_y + finish_line_height / 2.0
            # 起点到终点的直线距离（按本回合的起点）
            start_x, start_y = env.START_POS
            start_to_finish_dist = math.hypot(finish_cx - start_x, finish_cy - start_y)
            # env.reset() # reset_env() 会调用 reset，无需重复

            score = 0.0
            done = False
            observation = env.reset_env()
            episode_loss = []
            step_count = 0
            last_reward = 0.0
            ep_start = time.time()

            while not done:
                if render:
                    clock.tick(FPS)
                    # 刷新赛道+车（仅渲染，训练不依赖）
                    win.blit(track_surface, (0, 0))
                    env.draw(win)
                    pygame.display.update()
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            done = True
                            break
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
            loss_history.append(loss_mean)
            scores.append(score)

            # 成功判定：终点标志或最后一步为终点奖励
            succeeded = getattr(env, 'is_finished', False) or (last_reward >= success_reward - 1e-6)
            took = time.time() - ep_start

            # 计算碰撞点到终点的直线距离（若本回合发生了碰撞）
            crash_dist_val = None
            if getattr(env, 'is_collide', False):
                crash_dist_val = math.hypot(finish_cx - env.x, finish_cy - env.y)
            crash_dist_str = f"{crash_dist_val:.1f}" if crash_dist_val is not None else "N/A"
            # 终局最终位置与终点的直线距离（无论何种终止）
            final_dist = math.hypot(finish_cx - env.x, finish_cy - env.y)
            term_reason = getattr(env, 'termination_reason', 'unknown')

            if succeeded:
                attempts = tries_since_last_success + 1  # 包含本次成功
                success_count += 1
                attempts_list.append(attempts)
                success_history.append(1)
                tries_note = f"attempts={attempts} (fails {tries_since_last_success}+1)"
                tries_since_last_success = 0
                status = "SUCCESS"
                log_print(log_f,
                    f">>> SUCCESS #{success_count} | ep {episode_idx} | score {score:.2f} | "
                    f"last_reward {last_reward:.1f} | loss {loss_mean:.3f} | eps {eps_value:.3f} | "
                    f"steps {step_count} | start_dist {start_to_finish_dist:.1f} | crash_dist {crash_dist_str} | "
                    f"final_dist {final_dist:.1f} | term {term_reason} | "
                    f"{tries_note} | {took*1000:.0f} ms"
                )
            else:
                tries_since_last_success += 1
                success_history.append(0)
                status = "fail"
                tries_note = f"fails_since_last={tries_since_last_success}"

            # checkpoint: 周期性保存
            now = time.time()
            formatted_time = time.strftime("%Y%m%d%H%M%S", time.localtime(now))
            if now - last_ckpt_time >= ckpt_interval_sec:
                ckpt_path = os.path.join(ckpt_dir, f"ckpt_time_{int(formatted_time)}.pt")
                try:
                    agent.save_model(ckpt_path)
                    log_print(log_f, f"[ckpt] saved periodic checkpoint -> {ckpt_path}")
                except Exception as e:
                    log_print(log_f, f"[ckpt] save failed: {e}")
                last_ckpt_time = now
            # checkpoint: 成功时保存
            if succeeded:
                ckpt_path = os.path.join(ckpt_dir, f"success_{success_count:03d}_ep_{episode_idx}.pt")
                try:
                    agent.save_model(ckpt_path)
                    log_print(log_f, f"[ckpt] saved success checkpoint -> {ckpt_path}")
                except Exception as e:
                    log_print(log_f, f"[ckpt] save failed: {e}")

            if (episode_idx % print_every) == 0:
                log_print(log_f,
                    f"ep {episode_idx:5d} | {status:7s} | score {score:9.2f} | "
                    f"last_reward {last_reward:6.1f} | loss {loss_mean:7.3f} | eps {eps_value:5.3f} | "
                    f"steps {step_count:4d} | start_dist {start_to_finish_dist:.1f} | crash_dist {crash_dist_str} | "
                    f"final_dist {final_dist:.1f} | reason: {term_reason} | "
                    f"{tries_note} | {took*1000:.0f} ms"
                )

            # 曲线（按 v3 风格）
            all_curve_data.append(f"v4,reward,{episode_idx},{score}\n")
            all_curve_data.append(f"v4,loss,{episode_idx},{loss_mean}\n")
            all_curve_data.append(f"v4,epsilon,{episode_idx},{eps_value}\n")
            # success_rate 的滑窗在写文件时统一计算（这里追加单点标记）
            episode_idx += 1

    # 计算 success_rate 序列（窗口 20）
    success_rate_series = []
    window = 20
    for idx in range(len(success_history)):
        rate = sum(success_history[max(0, idx - window + 1): idx + 1]) / min(idx + 1, window)
        success_rate_series.append(rate)
        all_curve_data.append(f"v4,success_rate,{idx},{rate}\n")

    # 汇总
    episodes = len(scores)
    avg_score = sum(scores) / episodes if episodes else 0.0
    std_score = statistics.stdev(scores) if episodes > 1 else 0.0
    avg_loss = float(np.mean(loss_history)) if loss_history else 0.0
    avg_eps = float(np.mean(epsilon_history)) if epsilon_history else agent.epsilon
    avg_tries = float(np.mean(attempts_list)) if attempts_list else 0.0
    std_tries = statistics.stdev(attempts_list) if len(attempts_list) > 1 else 0.0
    succ_rate = (sum(success_history) / episodes) if episodes > 0 else 0.0
    took_total = time.time() - t0
    epm = episodes / (took_total / 60.0) if took_total > 0 else 0.0

    summary_block = (
        "==== Summary (v4 new track) ====\n"
        f"episodes: {episodes}\n"
        f"successes: {sum(success_history)}\n"
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

    # 写文件
    with open(output_curve, 'w') as f:
        f.writelines(all_curve_data)
    with open(output_summary, 'w') as f:
        f.write(summary_block)

    print("\n" + summary_block)
    print(f"Curve data saved to {output_curve}")
    print(f"Summary saved to {output_summary}")
    pygame.quit()


if __name__ == '__main__':
    # 直接运行：默认无渲染训练
    run_v4_training(
        max_success=100,
        batch_size=128,
        max_mem_size=100000,
        windowless=True,
        render=False,
        # 如需渲染调试：render=True, headless=False
        # max_episodes=4000,
    )