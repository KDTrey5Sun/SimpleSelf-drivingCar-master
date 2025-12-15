from __future__ import annotations

import os
import time
import random
import pygame
import numpy as np
import statistics
import math
import csv
from pathlib import Path
from DQN import Agent
from concurrent.futures import ProcessPoolExecutor, as_completed  # 并行


class OnlineVariance:
    """单通道增量方差统计，用于减少多次遍历大数组的开销。"""
    __slots__ = ('count', 'mean', 'M2')

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, value: float | int) -> None:
        val = float(value)
        self.count += 1
        delta = val - self.mean
        self.mean += delta / self.count
        delta2 = val - self.mean
        self.M2 += delta * delta2

    def variance(self) -> float:
        if self.count <= 1:
            return 0.0
        return self.M2 / self.count

    def mean_value(self) -> float:
        return self.mean if self.count > 0 else 0.0

# ==========================
# 在文件内配置实验参数（无需命令行）
# ==========================
# 在这里直接写入要对比的 replay buffer size 列表
# 留空列表 [] 时将执行单次训练（使用下方 TRAIN_KWARGS 中的默认 max_mem_size）
BUFFER_SIZES = [100000, 50000, 10000, 5000, 1000]

# 每个 size 重复次数
REPEATS = 50

# Sweep 标签（会作为输出子目录名的一部分），可设为 None
TAG = 'rb122k'

# 基础随机种子（不同重复会在此基础上依次递增）
SHARED_SEED = 123

# 训练公共参数（可按需调整）
TRAIN_KWARGS = {
    'max_success': 100,
    'max_episodes': 10000,
    'batch_size': 128,
    'gamma': 0.99,
    'lr': 5e-4,
    'init_epsilon': 1.0,
    'eps_end': 0.05,
    'eps_dec': 1e-4,
    'learn_starts': 5000,
    # 更稳的目标网络更新：硬更新间隔调大，或配合软更新系数（见下）
    'replace_target': 5000,
    # 软更新系数（>0 开启软更新；Agent 内部需支持），建议 0.005 左右
    'target_soft_tau': 0.005,
    'success_reward': 3000,
    'windowless': True,
    'render': False,
    'print_every': 1,
    'convergence_threshold': 0.7,
    'convergence_patience': 15,
    'convergence_min_episodes': 200,
    'enable_early_stop': True,
}

def run_v5_training(
    max_success=100,
    batch_size=128,
    max_mem_size=100000,
    gamma=0.99,
    lr=5e-4,
    init_epsilon=1.0,
    eps_end=0.05,
    eps_dec=1e-4,
    learn_starts=5000,
    replace_target=5000,
    success_reward=3000,
    windowless=True,
    render=False,
    output_curve='./v5_faster/v5_exp_data/curve_data_v5.txt',
    output_summary='./v5_faster/v5_exp_data/summary_v5.txt',
    output_log='./v5_faster/v5_exp_data/train_log_v5.txt',
    print_every=1,
    max_episodes=None,
    seed=None,
    # 软更新开关与参数（Agent 需支持）
    target_soft_tau: float = 0.0,
    # 运行标识（用于在 summary 开头显示）
    run_size: int | None = None,
    run_rep: int | None = None,
    # 收敛相关（早停已移除，仅用于离线统计）
    convergence_threshold: float = 0.7,
    convergence_patience: int = 15,
    convergence_min_episodes: int = 200,
    enable_early_stop: bool = False,
):
    # 头less模式屏蔽窗口/音频
    if windowless and not render:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    # 随机种子（保证可复现）
    if seed is None:
        seed = int(os.environ.get("SEED", "123"))
    random.seed(seed)
    np.random.seed(seed)

    # 输出目录
    for p in [output_curve, output_summary, output_log]:
        d = os.path.dirname(os.path.abspath(p))
        os.makedirs(d, exist_ok=True)

    # 延迟导入赛道/小车模块
    import autocar_v5 as ac

    pygame.init()
    win = pygame.display.set_mode((max(1, ac.WIDTH), max(1, ac.HEIGHT))) if render else pygame.display.set_mode((1, 1))
    if render:
        pygame.display.set_caption("DQN Car v5 - 训练")
        images = [(ac.GRASS, (0, 0)), (ac.TRACK, (0, 0)), (ac.FINISH, ac.FINISH_POSITION), (ac.TRACK_BORDER, (0, 0))]
    clock = pygame.time.Clock()

    def draw_frame():
        for img, pos in images:
            win.blit(img, pos)
        env.draw(win)
        pygame.display.update()

    def log_print(fh, msg):
        print(msg)
        fh.write(msg + "\n")
        fh.flush()

    def ensure_parent_dir(file_path: str):
        try:
            d = os.path.dirname(os.path.abspath(file_path))
            if d:
                os.makedirs(d, exist_ok=True)
        except Exception:
            pass

    # 代理
    agent = Agent(
        gamma=gamma,
        epsilon=init_epsilon,
        batch_size=batch_size,
        n_actions=4,
        eps_end=eps_end,
        input_dims=5,
        lr=lr,
        max_mem_size=max_mem_size,
        eps_dec=eps_dec,
        combined=False,
        learn_starts=learn_starts,
        replace_target=replace_target,
        clip_reward=False,
        target_soft_tau=target_soft_tau,
    )

    # 统计
    scores, loss_history, epsilon_history, success_history = [], [], [], []
    update_loss_history = []            # 每次 learn() 调用的即时 loss
    td_mean_history = []                # 每次 learn() 的 |TD error| 均值
    td_std_history = []                 # 每次 learn() 的 |TD error| 标准差
    # all_curve_data = []  # 注释：不再收集/写入曲线数据文件
    success_count = 0
    tries_since_last_success = 0
    attempts_list = []
    episode_idx = 0
    t0 = time.time()
    total_samples_collected = 0

    first_success_episode = None
    samples_at_first_success = None
    # samples_at_first_success: 首次成功时的累计样本数
    samples_at_epsilon_min = None
    learning_steps_to_first_success = None
    # learning_steps_to_first_success: 首次成功前一共调用了多少次 learn()。
    learn_steps_counter = 0
    # 新增：每个 episode 结束时的累计样本计数，用于 samples_to_convergence 计算
    cumulative_samples = []
    # 动态 epsilon：记录基础 eps_dec，控制冻结/恢复
    base_eps_dec = getattr(agent, 'eps_dec', 0.0)
    # 增量统计 & 早停控制
    reward_stats = OnlineVariance()
    loss_stats = OnlineVariance()
    success_rate_stats = OnlineVariance()
    # 早停相关变量移除

    # ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(output_log)), 'checkpoints')
    # os.makedirs(ckpt_dir, exist_ok=True)
    # last_ckpt_time = time.time()
    # ckpt_interval_sec = 600

    with open(output_log, 'w') as log_f:
        log_print(log_f, f"==== Start v5 training | batch_size={batch_size}, replay={max_mem_size} | seed={seed} ====")

        while success_count < max_success:
            if max_episodes is not None and episode_idx >= max_episodes:
                log_print(log_f, f"reach max_episodes={max_episodes}, stop.")
                break

            env = ac.ComputerCar(max_vel=400, rotation_vel=4)
            start_x, start_y = env.START_POS

            score = 0.0
            done = False
            observation = env.reset_env()
            episode_loss = []
            step_count = 0
            last_reward = 0.0
            ep_start = time.time()

            ring_cx, ring_cy = ac.WIDTH / 2.0, ac.HEIGHT / 2.0
            prog_thresh = getattr(ac, 'PROGRESS_THRESH', 0.01)
            angle_sum = 0.0
            center_sum = 0.0
            align_sum = 0.0
            np_streak = 0
            np_streak_max = 0
            last_x, last_y = env.x, env.y
            prev_sx, prev_sy = None, None
            CURVE_DETECT_THRESH = getattr(ac, 'CURVE_DETECT_THRESH', 0.012)

            while not done:
                if render:
                    clock.tick(ac.FPS)
                    draw_frame()
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            done = True
                            break
                else:
                    pygame.event.pump()

                # 线性衰减由 Agent.learn() 内部基于 eps_dec 自动处理

                action = agent.choose_action(observation)
                observation_, reward, done = env.step(action)
                last_reward = reward
                score += reward
                agent.memory.store_transition(observation, action, reward, observation_, done)
                total_samples_collected += 1
                observation = observation_

                loss = agent.learn()
                if loss is not None:
                    episode_loss.append(loss)
                    update_loss_history.append(loss)
                    loss_stats.update(loss)
                    td_mean, td_std = agent.get_last_td_stats()
                    if td_mean is not None:
                        td_mean_history.append(td_mean)
                    if td_std is not None:
                        td_std_history.append(td_std)
                    learn_steps_counter += 1
                    if samples_at_epsilon_min is None and getattr(agent, 'epsilon', None) is not None:
                        try:
                            if agent.epsilon <= getattr(agent, 'eps_min', 0.0) + 1e-12:
                                samples_at_epsilon_min = total_samples_collected
                        except Exception:
                            pass
                step_count += 1

                dx = env.x - last_x
                dy = env.y - last_y
                theta_prev = math.atan2(ring_cy - last_y, last_x - ring_cx)
                theta_cur = math.atan2(ring_cy - env.y, env.x - ring_cx)
                dtheta = (theta_cur - theta_prev + math.pi) % (2 * math.pi) - math.pi
                angle_progress = max(0.0, dtheta)
                angle_sum += angle_progress

                try:
                    border_dist = env.get_distance_to_border()
                    center_offset = abs(ac.MID_TRACK - border_dist)
                    center_factor = 1.0 - min(1.0, center_offset / max(ac.MID_TRACK, 1e-6))
                except Exception:
                    center_factor = 0.0
                center_sum += center_factor

                step_norm = math.hypot(dx, dy) or 1e-6
                sx, sy = (dx / step_norm, dy / step_norm)
                tx, ty = -math.sin(theta_cur), -math.cos(theta_cur)
                align_curve = abs(sx * tx + sy * ty)
                if prev_sx is None or prev_sy is None:
                    align_straight = align_curve
                else:
                    align_straight = abs(sx * prev_sx + sy * prev_sy)
                w_curve = min(1.0, abs(dtheta) / max(1e-9, CURVE_DETECT_THRESH))
                align_factor = w_curve * align_curve + (1.0 - w_curve) * align_straight
                align_sum += align_factor

                if angle_progress < prog_thresh:
                    np_streak += 1
                    if np_streak > np_streak_max:
                        np_streak_max = np_streak
                else:
                    np_streak = 0

                last_x, last_y = env.x, env.y
                prev_sx, prev_sy = sx, sy

            eps_value = agent.epsilon
            loss_mean = float(np.mean(episode_loss) if episode_loss else 0.0)
            epsilon_history.append(eps_value)
            loss_history.append(loss_mean)
            scores.append(score)
            reward_stats.update(score)

            succeeded = getattr(env, 'is_finished', False) or (last_reward >= success_reward - 1e-6)
            took = time.time() - ep_start
            term_reason = getattr(env, 'termination_reason', 'unknown')

            crash_ang_str = "N/A"
            try:
                if term_reason == 'collision' or getattr(env, 'is_collide', False):
                    theta_start = math.atan2(ring_cy - start_y, start_x - ring_cx)
                    theta_crash = math.atan2(ring_cy - env.y, env.x - ring_cx)
                    dtheta = (theta_crash - theta_start + math.pi) % (2 * math.pi) - math.pi
                    crash_ang_str = f"{math.degrees(dtheta):.2f}"
                else:
                    crash_ang_str = "N/A"
            except Exception:
                crash_ang_str = "N/A"

            if succeeded:
                attempts = tries_since_last_success + 1
                success_count += 1
                attempts_list.append(attempts)
                success_history.append(1)
                tries_note = f"attempts={attempts} (fails {tries_since_last_success}+1)"
                tries_since_last_success = 0
                status = "SUCCESS"
                if first_success_episode is None:
                    first_success_episode = episode_idx
                    samples_at_first_success = total_samples_collected
                    learning_steps_to_first_success = learn_steps_counter
                    # 保持默认线性衰减，无需在成功后重设 eps_dec
                log_print(log_f,
                    f">>> SUCCESS #{success_count} | ep {episode_idx} | score {score:.2f} | "
                    f"last_reward {last_reward:.1f} | loss {loss_mean:.3f} | eps {eps_value:.3f} | "
                    f"steps {step_count} | term {term_reason} | crash_angle_deg {crash_ang_str} | "
                    f"{tries_note} | {took*1000:.0f} ms"
                )
            else:
                tries_since_last_success += 1
                success_history.append(0)
                status = "fail"
                tries_note = f"fails_since_last={tries_since_last_success}"

            # now = time.time()
            # formatted_time = time.strftime("%Y%m%d%H%M%S", time.localtime(now))
            # if now - last_ckpt_time >= ckpt_interval_sec:
            #     ckpt_path = os.path.join(ckpt_dir, f"ckpt_time_{int(formatted_time)}.pt")
            #     try:
            #         os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            #         agent.save_model(ckpt_path)
            #         log_print(log_f, f"[ckpt] saved periodic checkpoint -> {ckpt_path}")
            #     except Exception as e:
            #         log_print(log_f, f"[ckpt] save failed: {e}")
            #     last_ckpt_time = now
            # if succeeded:
            #     ckpt_path = os.path.join(ckpt_dir, f"success_{success_count:03d}_ep_{episode_idx}.pt")
            #     try:
            #         os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            #         agent.save_model(ckpt_path)
            #         log_print(log_f, f"[ckpt] saved success checkpoint -> {ckpt_path}")
            #     except Exception as e:
            #         log_print(log_f, f"[ckpt] save failed: {e}")

            if (episode_idx % print_every) == 0:
                log_print(log_f,
                    f"ep {episode_idx:5d} | {status:7s} | score {score:9.2f} | "
                    f"last_reward {last_reward:6.1f} | loss {loss_mean:7.3f} | epsilon {eps_value:5.3f} | "
                    f"steps {step_count:4d} | reason: {term_reason} | crash_angle_deg {crash_ang_str} | "
                    f"{tries_note} | {took*1000:.0f} ms"
                )
            recent_window = success_history[-100:]
            recent_success_rate = (sum(recent_window) / len(recent_window)) if recent_window else 0.0

            # 注释：不再向曲线数据缓冲追加记录
            # all_curve_data.append(f"v5,reward,{episode_idx},{score}\n")
            # all_curve_data.append(f"v5,loss,{episode_idx},{loss_mean}\n")
            # all_curve_data.append(f"v5,epsilon,{episode_idx},{eps_value}\n")
            # if td_mean_history:
            #     all_curve_data.append(f"v5,td_mean,{episode_idx},{td_mean_history[-1]}\n")
            # if td_std_history:
            #     all_curve_data.append(f"v5,td_std,{episode_idx},{td_std_history[-1]}\n")
            episode_idx += 1
            # 记录本回合结束时累计采样数
            cumulative_samples.append(total_samples_collected)

            # 不再依据早停中断训练

    # success_rate 计算方式修改：
    #  - 若当前总回合数 <= 100：使用 [0 .. idx] 的累计成功率 (累计成功次数 / 当前回合数)
    #  - 若当前总回合数 > 100：使用最近 100 个回合的窗口成功率 (最近100回合成功次数 / 100)
    # 统一改为“最近100回合成功率”窗口（不足100则使用已有回合数）
    success_rate_series = []
    for idx in range(len(success_history)):
        start = max(0, idx - 99)
        window = success_history[start: idx + 1]
        denom = len(window)
        rate = (sum(window) / denom) if denom > 0 else 0.0
        success_rate_series.append(rate)
        success_rate_stats.update(rate)
        # all_curve_data.append(f"v5,success_rate,{idx},{rate}\n")  # 注释：不再写曲线成功率

    # ===== 收敛指标：首次窗口成功率 >= 0.7 =====
    episodes_to_convergence = None
    samples_to_convergence = None
    for i, r in enumerate(success_rate_series):
        if r >= 0.7:
            episodes_to_convergence = i
            if i < len(cumulative_samples):
                samples_to_convergence = cumulative_samples[i]
            break

    episodes = len(scores)
    # ====== 稳定性附加指标 ======
    def _var(arr):
        return float(np.var(arr, ddof=0)) if len(arr) > 1 else 0.0
    def _tail(arr, k=100):
        if not arr: return []
        return arr[-k:]
    update_loss_variance = loss_stats.variance()
    update_loss_variance_last100 = _var(_tail(update_loss_history))
    td_mean_last100 = float(np.mean(_tail(td_mean_history))) if td_mean_history else 0.0
    td_std_last100 = float(np.mean(_tail(td_std_history))) if td_std_history else 0.0
    avg_score = sum(scores) / episodes if episodes else 0.0
    std_score = statistics.stdev(scores) if episodes > 1 else 0.0
    avg_loss = float(np.mean(loss_history)) if loss_history else 0.0
    avg_eps = float(np.mean(epsilon_history)) if epsilon_history else agent.epsilon
    avg_tries = float(np.mean(attempts_list)) if attempts_list else 0.0
    std_tries = statistics.stdev(attempts_list) if len(attempts_list) > 1 else 0.0
    # Summary 成功率改为“最近100回合成功率”
    if episodes > 0:
        recent_window = success_history[-100:]
        succ_rate = sum(recent_window) / len(recent_window)
    else:
        succ_rate = 0.0
    took_total = time.time() - t0
    epm = episodes / (took_total / 60.0) if took_total > 0 else 0.0

    # ===== 额外稳定性方差指标 =====
    reward_variance_full = reward_stats.variance()
    reward_variance_last100 = _var(_tail(scores))
    success_rate_variance_full = success_rate_stats.variance()
    success_rate_variance_last100 = _var(_tail(success_rate_series))
    loss_variance_full = update_loss_variance
    loss_variance_last100 = update_loss_variance_last100

    convergence_speed = (1.0 / episodes_to_convergence) if (episodes_to_convergence is not None and episodes_to_convergence > 0) else 0.0

    buffer_capacity = int(getattr(agent.memory, 'mem_size', max_mem_size))
    final_buffer_occupancy = int(min(getattr(agent.memory, 'mem_cntr', total_samples_collected), buffer_capacity))
    samples_dropped = int(max(0, total_samples_collected - buffer_capacity))
    effective_learn_starts = int(max(learn_starts, batch_size))
    avg_samples_per_episode = float(total_samples_collected / episodes) if episodes > 0 else 0.0
    # 使用 memory 中的覆盖计数（若存在）
    overwritten_count = int(getattr(agent.memory, 'overwritten_count', samples_dropped))

    time_sec = float(took_total)

    # 在 summary 开头显示 size / rep，未提供时回退为 max_mem_size / N/A
    _hdr_size = int(run_size) if run_size is not None else int(max_mem_size)
    _hdr_rep = (str(int(run_rep)) if run_rep is not None else 'N/A')
    summary_block = (
        f"==== Summary (v5) | size={_hdr_size} | rep={_hdr_rep} ====\n"
        f"episodes: {episodes}\n"
        f"successes: {sum(success_history)}\n"
        f"success_rate: {succ_rate:.3f}\n"
        f"avg_score: {avg_score:.2f}\n"
        f"std_score: {std_score:.2f}\n"
        f"avg_loss: {avg_loss:.3f}\n"
        f"avg_epsilon: {avg_eps:.3f}\n"
        f"avg_tries: {avg_tries:.2f}\n"
        f"tries_std: {std_tries:.2f}\n"
        f"total_samples_collected: {total_samples_collected}\n"
        f"avg_samples_per_episode: {avg_samples_per_episode:.2f}\n"
        f"first_success_episode: {first_success_episode if first_success_episode is not None else 'N/A'}\n"
        f"samples_at_first_success: {samples_at_first_success if samples_at_first_success is not None else 'N/A'}\n"
        f"episodes_to_convergence: {episodes_to_convergence if episodes_to_convergence is not None else 'N/A'}\n"
        f"samples_to_convergence: {samples_to_convergence if samples_to_convergence is not None else 'N/A'}\n"
        f"convergence_speed: {convergence_speed:.6f}\n"
        f"eps_min_reached_at_samples: {samples_at_epsilon_min if samples_at_epsilon_min is not None else 'N/A'}\n"
        f"buffer_capacity: {buffer_capacity}\n"
        f"final_buffer_occupancy: {final_buffer_occupancy}\n"
        f"samples_dropped_overwritten: {samples_dropped}\n"
        f"overwritten_count: {overwritten_count}\n"
        f"effective_learn_starts: {effective_learn_starts}\n"
        f"time_min: {took_total/60:.2f}\n"
        f"episodes_per_min: {epm:.1f}\n"
        f"time_sec: {time_sec:.2f}\n"
        f"update_loss_variance: {update_loss_variance:.6f}\n"
        f"update_loss_variance_last100: {update_loss_variance_last100:.6f}\n"
        f"td_mean_last100: {td_mean_last100:.6f}\n"
        f"td_std_last100: {td_std_last100:.6f}\n"
        f"learning_steps_to_first_success: {learning_steps_to_first_success if learning_steps_to_first_success is not None else 'N/A'}\n"
        f"reward_variance_full: {reward_variance_full:.6f}\n"
        f"reward_variance_last100: {reward_variance_last100:.6f}\n"
        f"success_rate_variance_full: {success_rate_variance_full:.6f}\n"
        f"success_rate_variance_last100: {success_rate_variance_last100:.6f}\n"
        f"loss_variance_full: {loss_variance_full:.6f}\n"
        f"loss_variance_last100: {loss_variance_last100:.6f}\n"
        # 早停信息移除
        f"convergence_threshold: {convergence_threshold:.3f}\n"
        f"convergence_patience: {convergence_patience}\n"
        f"convergence_min_episodes: {convergence_min_episodes}\n"
    )

    # 注释：不再生成和写入 curve_data.txt
    # ensure_parent_dir(output_curve)
    # with open(output_curve, 'w') as f:
    #     f.writelines(all_curve_data)
    ensure_parent_dir(output_summary)
    with open(output_summary, 'w') as f:
        f.write(summary_block)

    print("\n" + summary_block)
    # print(f"Curve data saved to {output_curve}")  # 注释：不再提示曲线数据保存
    print(f"Summary saved to {output_summary}")
    pygame.quit()

    return {
        'episodes': episodes,
        'successes': int(sum(success_history)),
        'success_rate': float(succ_rate),
        'avg_score': float(avg_score),
        'std_score': float(std_score),
        'avg_loss': float(avg_loss),
        'avg_epsilon': float(avg_eps),
        'avg_tries': float(avg_tries),
        'tries_std': float(std_tries),
        'time_min': float(took_total/60),
        'episodes_per_min': float(epm),
        'time_sec': time_sec,
        'total_samples_collected': int(total_samples_collected),
        'avg_samples_per_episode': float(avg_samples_per_episode),
        'first_success_episode': int(first_success_episode) if first_success_episode is not None else None,
        'samples_at_first_success': int(samples_at_first_success) if samples_at_first_success is not None else None,
        'episodes_to_convergence': int(episodes_to_convergence) if episodes_to_convergence is not None else None,
        'samples_to_convergence': int(samples_to_convergence) if samples_to_convergence is not None else None,
        'convergence_speed': float(convergence_speed),
        'eps_min_reached_at_samples': int(samples_at_epsilon_min) if samples_at_epsilon_min is not None else None,
        'buffer_capacity': int(buffer_capacity),
        'final_buffer_occupancy': int(final_buffer_occupancy),
        'samples_dropped_overwritten': int(samples_dropped),
        'overwritten_count': overwritten_count,
        'effective_learn_starts': int(effective_learn_starts),
        'update_loss_variance': update_loss_variance,
        'update_loss_variance_last100': update_loss_variance_last100,
        'td_mean_last100': td_mean_last100,
        'td_std_last100': td_std_last100,
        'learning_steps_to_first_success': int(learning_steps_to_first_success) if learning_steps_to_first_success is not None else None,
        'reward_variance_full': reward_variance_full,
        'reward_variance_last100': reward_variance_last100,
        'success_rate_variance_full': success_rate_variance_full,
        'success_rate_variance_last100': success_rate_variance_last100,
        'loss_variance_full': loss_variance_full,
        'loss_variance_last100': loss_variance_last100,
        # 早停相关键移除，仅保留阈值配置供外部参考
        'convergence_threshold': float(convergence_threshold),
        'convergence_patience': int(convergence_patience),
        'convergence_min_episodes': int(convergence_min_episodes),
        'max_mem_size': int(max_mem_size),
        'batch_size': int(batch_size),
        'gamma': float(gamma),
        'lr': float(lr),
        'init_epsilon': float(init_epsilon),
        'eps_end': float(eps_end),
        'eps_dec': float(eps_dec),
        'learn_starts': int(learn_starts),
        'replace_target': int(replace_target),
        'success_reward': float(success_reward),
        'seed': int(seed),
        'render': bool(render),
        'windowless': bool(windowless),
        'output_curve': str(output_curve),
        'output_summary': str(output_summary),
        'output_log': str(output_log),
    }

# 顶层 worker，供进程池调用（可被 pickle）
def sweep_worker(task: dict) -> dict:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("FORCE_CPU", "1")
    res = run_v5_training(
        max_mem_size=task['size'],
        output_curve=task['curve_path'],
        output_summary=task['summary_path'],
        output_log=task['log_path'],
        seed=task['seed'],
        run_size=task['size'],
        run_rep=task['rep'],
        **task['run_kwargs'],
    )
    row = {
        'size': task['size'],
        'rep': task['rep'],
        'episodes': res['episodes'],
        'successes': res['successes'],
        'success_rate': res['success_rate'],
        'avg_score': res['avg_score'],
        'std_score': res['std_score'],
        'avg_loss': res['avg_loss'],
        'avg_epsilon': res['avg_epsilon'],
        'avg_tries': res['avg_tries'],
        'tries_std': res['tries_std'],
        'time_min': res['time_min'],
        'time_sec': res['time_sec'],
        'episodes_per_min': res['episodes_per_min'],
        'total_samples_collected': res['total_samples_collected'],
        'avg_samples_per_episode': res['avg_samples_per_episode'],
        'first_success_episode': res['first_success_episode'],
        'samples_at_first_success': res['samples_at_first_success'],
        'episodes_to_convergence': res['episodes_to_convergence'],
        'samples_to_convergence': res['samples_to_convergence'],
        'convergence_speed': res['convergence_speed'],
        'eps_min_reached_at_samples': res['eps_min_reached_at_samples'],
        'buffer_capacity': res['buffer_capacity'],
        'final_buffer_occupancy': res['final_buffer_occupancy'],
        'samples_dropped_overwritten': res['samples_dropped_overwritten'],
        'overwritten_count': res['overwritten_count'],
        'effective_learn_starts': res['effective_learn_starts'],
        'output_summary': res['output_summary'],
        'reward_variance_full': res['reward_variance_full'],
        'reward_variance_last100': res['reward_variance_last100'],
        'success_rate_variance_full': res['success_rate_variance_full'],
        'success_rate_variance_last100': res['success_rate_variance_last100'],
        'loss_variance_full': res['loss_variance_full'],
        'loss_variance_last100': res['loss_variance_last100'],
    }
    return row

def sweep_replay_buffer_sizes(
    buffer_sizes,
    repeats=1,
    base_output_dir='./v5_faster/v5_exp_data/replay_sweep',
    tag=None,
    shared_seed=123,
    **shared_kwargs,
):
    """对不同 replay buffer size 进行横向实验并生成 CSV 汇总（多进程并行），并输出每个 size 的均值/标准差。"""
    base_dir = Path(base_output_dir)
    if tag:
        base_dir = base_dir / str(tag)
    base_dir.mkdir(parents=True, exist_ok=True)

    csv_path = base_dir / 'sweep_summary.csv'
    fieldnames = [
        'size', 'rep', 'episodes', 'successes', 'success_rate', 'avg_score', 'std_score',
        'avg_loss', 'avg_epsilon', 'avg_tries', 'tries_std', 'time_min', 'time_sec', 'episodes_per_min',
        'total_samples_collected', 'avg_samples_per_episode', 'first_success_episode', 'samples_at_first_success',
        'episodes_to_convergence', 'samples_to_convergence', 'convergence_speed',
        'eps_min_reached_at_samples', 'buffer_capacity', 'final_buffer_occupancy', 'samples_dropped_overwritten',
        'effective_learn_starts',
        'reward_variance_full', 'reward_variance_last100', 'success_rate_variance_full', 'success_rate_variance_last100',
        'loss_variance_full', 'loss_variance_last100', 'overwritten_count', 'output_summary'
    ]

    # 构建任务列表
    tasks = []
    for size in buffer_sizes:
        for rep in range(1, repeats + 1):
            out_dir = base_dir / f'size_{int(size)}' / f'rep_{rep}'
            out_dir.mkdir(parents=True, exist_ok=True)
            curve_path = str(out_dir / 'curve_data.txt')
            summary_path = str(out_dir / 'summary.txt')
            log_path = str(out_dir / 'train_log.txt')

            run_kwargs = dict(shared_kwargs)
            # 安全约束，避免 learn_starts/batch_size 大于 buffer
            if run_kwargs.get('learn_starts', None) is not None and run_kwargs['learn_starts'] > int(size):
                run_kwargs['learn_starts'] = int(size)
            if run_kwargs.get('batch_size', None) is not None and run_kwargs['batch_size'] > int(size):
                run_kwargs['batch_size'] = int(size)

            seed = shared_seed + rep - 1
            task = {
                'size': int(size),
                'rep': rep,
                'out_dir': str(out_dir),
                'curve_path': curve_path,
                'summary_path': summary_path,
                'log_path': log_path,
                'seed': seed,
                'run_kwargs': run_kwargs,
            }
            tasks.append(task)

    # 并行度：SWEEP_WORKERS > SLURM_CPUS_PER_TASK > os.cpu_count()
    def _auto_workers():
        for k in ('SWEEP_WORKERS', 'SLURM_CPUS_PER_TASK'):
            v = os.getenv(k)
            if v and v.isdigit() and int(v) > 0:
                return int(v)
        return max(1, (os.cpu_count() or 1))

    max_workers = _auto_workers()
    print(f"[Sweep] total tasks={len(tasks)}, parallel workers={max_workers}")

    results = []
    if max_workers <= 1:
        for t in tasks:
            print(f"[Sweep] run serial: size={t['size']} rep={t['rep']}")
            results.append(sweep_worker(t))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            fut2task = {ex.submit(sweep_worker, t): t for t in tasks}
            for fut in as_completed(fut2task):
                t = fut2task[fut]
                try:
                    row = fut.result()
                    print(f"[Sweep] done: size={t['size']} rep={t['rep']}")
                    results.append(row)
                except Exception as e:
                    print(f"[Sweep] failed: size={t['size']} rep={t['rep']} err={e}")

    # 详细CSV（每次运行一行）
    results.sort(key=lambda r: (r['size'], r['rep']))
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Per-run summary CSV: {csv_path}")

    # 聚合：按 size 计算均值/标准差
    def _nanmean_std(vals):
        vals2 = [float(x) for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
        if len(vals2) == 0:
            return (None, None)
        if len(vals2) == 1:
            return (vals2[0], 0.0)
        return (statistics.mean(vals2), statistics.stdev(vals2))

    # 需要聚合的数值字段
    agg_numeric_fields = [
        'episodes', 'successes', 'success_rate', 'avg_score', 'std_score',
        'avg_loss', 'avg_epsilon', 'avg_tries', 'tries_std',
        'time_min', 'time_sec', 'episodes_per_min',
        'total_samples_collected', 'avg_samples_per_episode',
        'first_success_episode', 'samples_at_first_success',
        'episodes_to_convergence', 'samples_to_convergence', 'convergence_speed',
        'eps_min_reached_at_samples',
        'final_buffer_occupancy', 'samples_dropped_overwritten',
        'reward_variance_full', 'reward_variance_last100', 'success_rate_variance_full', 'success_rate_variance_last100', 'loss_variance_full', 'loss_variance_last100', 'overwritten_count',
        # 早停聚合字段移除
    ]

    # 输出聚合CSV
    agg_csv_path = base_dir / 'sweep_agg.csv'
    agg_header = ['size', 'n']
    for k in agg_numeric_fields:
        agg_header += [f'{k}_mean', f'{k}_std']

    # 分组
    from collections import defaultdict
    by_size = defaultdict(list)
    for r in results:
        by_size[int(r['size'])].append(r)

    with open(agg_csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=agg_header)
        w.writeheader()
        for size in sorted(by_size.keys()):
            rows = by_size[size]
            out = {'size': size, 'n': len(rows)}
            for k in agg_numeric_fields:
                mean, std = _nanmean_std([rr.get(k, None) for rr in rows])
                out[f'{k}_mean'] = f"{mean:.6f}" if mean is not None else ''
                out[f'{k}_std'] = f"{std:.6f}" if std is not None else ''
            w.writerow(out)
            # 简要打印关键均值
            try:
                m_score = out['avg_score_mean']; m_sr = out['success_rate_mean']; m_epm = out['episodes_per_min_mean']
                print(f"[Agg] size={size} n={len(rows)} | avg_score={m_score} | success_rate={m_sr} | ep/min={m_epm}")
            except Exception:
                pass

    print(f"Aggregated CSV: {agg_csv_path}")
    return str(agg_csv_path)

if __name__ == '__main__':
    if BUFFER_SIZES:
        print(f"[Config] Sweep sizes={BUFFER_SIZES}, repeats={REPEATS}, tag={TAG}, seed={SHARED_SEED}")
        sweep_replay_buffer_sizes(
            buffer_sizes=BUFFER_SIZES,
            repeats=REPEATS,
            base_output_dir='./v5_faster/v5_exp_data/replay_sweep',
            tag=TAG,
            shared_seed=SHARED_SEED,
            **TRAIN_KWARGS,
        )
    else:
        print("[Config] Single training run with TRAIN_KWARGS")
        run_v5_training(**TRAIN_KWARGS)