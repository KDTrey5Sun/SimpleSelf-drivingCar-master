import os
import time
import statistics
import math
import pygame
import numpy as np
from autocar_v4 import ComputerCar, FPS
from DQN import Agent
from draw_v4 import draw, draw_track, WIDTH, HEIGHT, TRACK_WIDTH, LANE_WIDTH, CENTER_X, TURN_RADIUS

"""
v4_show：可视化训练，并按照 v4 训练脚本的存储与日志方式输出：
- 训练日志：output_log
- 曲线数据：output_curve（reward/epsilon/loss/success_rate）
- 汇总：output_summary
- 周期性与成功时保存 checkpoint
"""

# 可调起点（与 v4 保持一致）
START_SHIFT_X = 200

# 输出路径（为避免与正式训练冲突，使用 _show 后缀）
OUTPUT_CURVE = './v4/v4_show_exp_data/curve_data_v4_show.txt'
OUTPUT_SUMMARY = './v4/v4_show_exp_data/summary_v4_show.txt'
OUTPUT_LOG = './v4/v4_show_exp_data/train_log_v4_show.txt'

# 成功阈值（与环境中的终点奖励一致）
SUCCESS_REWARD = 3000.0

# 打印频率与 checkpoint 配置
PRINT_EVERY = 1
CKPT_INTERVAL_SEC = 600  # 每 10 分钟保存一次

if __name__ == '__main__':
    # 确保输出目录存在
    for p in [OUTPUT_CURVE, OUTPUT_SUMMARY, OUTPUT_LOG]:
        d = os.path.dirname(os.path.abspath(p))
        os.makedirs(d, exist_ok=True)

    pygame.init()
    WIN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("DQN Car v4 - 90°弯道（可视化训练）")
    clock = pygame.time.Clock()

    # 预绘制赛道表面（用于渲染时复用）
    draw_track(WIN)
    TRACK_SURF = WIN.copy()

    # 环境
    env = ComputerCar(max_vel=8, rotation_vel=4)
    base_start_x = CENTER_X + 400
    base_start_y = HEIGHT // 2 + TURN_RADIUS - TRACK_WIDTH // 2
    env.START_POS = (base_start_x + START_SHIFT_X, base_start_y)
    env.reset()

    # Agent（与 v3 风格一致，简单稳定）
    agent = Agent(
        gamma=0.95,
        epsilon=1.0,
        batch_size=128,
        n_actions=4,
        eps_end=0.1,
        input_dims=4,
        lr=5e-4,
        max_mem_size=50000,
        eps_dec=0.002,
        combined=False
    )

    # 统计与日志
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

    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(OUTPUT_LOG)), 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    last_ckpt_time = time.time()

    run = True

    # 终点中心点
    finish_line_y = 40
    finish_line_height = 20
    finish_cx = CENTER_X
    finish_cy = finish_line_y + finish_line_height / 2.0

    def log_print(fh, msg):
        print(msg)
        fh.write(msg + "\n")
        fh.flush()

    with open(OUTPUT_LOG, 'w') as log_f:
        log_print(log_f, f"==== Start v4_show | batch_size={agent.batch_size}, replay={agent.memory.mem_size} ====")

        while run:
            # 每回合起点到终点距离（按当前 START_POS）
            start_x, start_y = env.START_POS
            start_to_finish_dist = math.hypot(finish_cx - start_x, finish_cy - start_y)

            score = 0.0
            done = False
            observation = env.reset_env()
            episode_loss = []
            step_count = 0
            last_reward = 0.0
            ep_start = time.time()

            while not done:
                clock.tick(FPS)

                # 渲染：赛道底图 + 车辆
                WIN.blit(TRACK_SURF, (0, 0))
                env.draw(WIN)  # 使用 env.draw 以保持一致
                pygame.display.update()

                # 处理关闭事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        run = False
                        done = True
                        break
                if not run:
                    break

                # 交互与学习
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

            # 记录并打印
            eps_value = agent.epsilon
            loss_mean = float(np.mean(episode_loss) if episode_loss else 0.0)
            epsilon_history.append(eps_value)
            loss_history.append(loss_mean)
            scores.append(score)

            # 成功判定：以环境标志或终点奖励为准
            succeeded = getattr(env, 'is_finished', False) or (last_reward >= SUCCESS_REWARD - 1e-6)
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
                attempts = tries_since_last_success + 1
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
                    f"final_dist {final_dist:.1f} | term {term_reason} | {tries_note} | {took*1000:.0f} ms"
                )
            else:
                tries_since_last_success += 1
                success_history.append(0)
                status = "fail"
                tries_note = f"fails_since_last={tries_since_last_success}"

            # checkpoint：周期性保存
            now = time.time()
            formatted_time = time.strftime("%Y%m%d%H%M%S", time.localtime(now))
            if now - last_ckpt_time >= CKPT_INTERVAL_SEC:
                ckpt_path = os.path.join(ckpt_dir, f"ckpt_time_{int(formatted_time)}.pt")
                try:
                    # v4 Agent 支持 save_model；若缺失可替代为 torch.save
                    agent.save_model(ckpt_path)  # type: ignore[attr-defined]
                    log_print(log_f, f"[ckpt] saved periodic checkpoint -> {ckpt_path}")
                except Exception as e:
                    log_print(log_f, f"[ckpt] save failed: {e}")
                last_ckpt_time = now
            # checkpoint：成功时保存
            if succeeded:
                ckpt_path = os.path.join(ckpt_dir, f"success_{success_count:03d}_ep_{episode_idx}.pt")
                try:
                    agent.save_model(ckpt_path)  # type: ignore[attr-defined]
                    log_print(log_f, f"[ckpt] saved success checkpoint -> {ckpt_path}")
                except Exception as e:
                    log_print(log_f, f"[ckpt] save failed: {e}")

            if (episode_idx % PRINT_EVERY) == 0:
                log_print(log_f,
                    f"ep {episode_idx:5d} | {status:7s} | score {score:9.2f} | "
                    f"last_reward {last_reward:6.1f} | loss {loss_mean:7.3f} | eps {eps_value:5.3f} | "
                    f"steps {step_count:4d} | start_dist {start_to_finish_dist:.1f} | crash_dist {crash_dist_str} | "
                    f"final_dist {final_dist:.1f} | reason: {term_reason} | {tries_note} | {took*1000:.0f} ms"
                )

            # 曲线（与 v4 一致的命名）
            all_curve_data.append(f"v4,reward,{episode_idx},{score}\n")
            all_curve_data.append(f"v4,loss,{episode_idx},{loss_mean}\n")
            all_curve_data.append(f"v4,epsilon,{episode_idx},{eps_value}\n")

            episode_idx += 1

            # 可视化版本无停止条件，按 ESC 退出或直接关闭窗口；也可添加上限
            # 这里演示：当达到若干成功后自动结束
            if success_count >= 5:
                run = False

    # success_rate（滑窗 20）
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
        "==== Summary (v4_show) ====\n"
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
    with open(OUTPUT_CURVE, 'w') as f:
        f.writelines(all_curve_data)
    with open(OUTPUT_SUMMARY, 'w') as f:
        f.write(summary_block)

    print("\n" + summary_block)
    print(f"Curve data saved to {OUTPUT_CURVE}")
    print(f"Summary saved to {OUTPUT_SUMMARY}")

    pygame.quit()