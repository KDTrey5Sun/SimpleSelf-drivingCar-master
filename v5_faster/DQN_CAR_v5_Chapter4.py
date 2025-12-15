import os
import time
import random
import pygame
import numpy as np
import statistics
import math
import csv
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


# ==========================
# RQ2: 批量大小敏感性分析 - 实验配置
# ==========================
# 研究问题：批量大小如何调节缓冲区大小对 DQN 性能的影响？
#
# SRQ2.1: 性能趋势一致性
#   - 指标: 跨批量大小的缓冲区排名 Spearman 秩相关
#   - 假设 H2.1: 缓冲区排名在不同批量下保持一致 (ρ > 0.7)
#
# SRQ2.2: 交互效应强度
#   - 指标: 双因素 ANOVA 交互项效应大小 η²
#   - 假设 H2.2: 批量与缓冲区弱交互 (η² < 0.1)
#
# 关键数据收集:
#   - success_rate: 成功率（主要性能指标）
#   - episodes_to_convergence: 首次成功的回合数（收敛速度）
#   - samples_to_convergence: 首次成功的样本数（样本效率）
#   - avg_score: 平均得分（整体性能）
#   - loss_variance_last100: 训练后期损失方差（稳定性）
# ==========================

# 在这里直接写入要对比的 replay buffer size 列表
# 留空列表 [] 时将执行单次训练（使用下方 TRAIN_KWARGS 中的默认 max_mem_size）
BUFFER_SIZES = [100000, 50000, 10000, 5000, 1000]

# 新增：批量大小 sweep 列表；若留空，则仅按 BUFFER_SIZES 单轴 sweep
BATCH_SIZES = [512, 128, 64, 32, 4]

# 每个 size 重复次数
REPEATS = 3

# Sweep 标签（会作为输出子目录名的一部分），可设为 None
TAG = 'rb133k'

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
    'replace_target': 5000,
    'target_soft_tau': 0.005,
    'success_reward': 3000,
    'windowless': True,
    'render': False,
    'print_every': 1,
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
    target_soft_tau=0.0,
    run_size=None,
    run_rep=None,
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
        """确保文件的父目录存在。"""
        try:
            d = os.path.dirname(os.path.abspath(file_path))
            if d:
                os.makedirs(d, exist_ok=True)
        except Exception:
            pass

    # 进程内存监控（与 Chapter3 一致，用于成本分析）
    process = None
    try:
        import psutil  # 延迟导入
        process = psutil.Process(os.getpid())
        mem_start_bytes = process.memory_info().rss
        mem_peak_bytes = mem_start_bytes
    except Exception:
        mem_start_bytes = None
        mem_peak_bytes = None

    def _update_mem_peak():
        nonlocal mem_peak_bytes
        if not process: return None
        try:
            rss = process.memory_info().rss
            if mem_peak_bytes is None or rss > mem_peak_bytes:
                mem_peak_bytes = rss
            return rss
        except Exception:
            return None

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
    scores = []
    loss_history = []
    epsilon_history = []
    success_history = []
    update_loss_history = []            # learn() 每次的即时 loss
    td_mean_history = []                # learn() 每次的 |TD| 均值
    td_std_history = []                 # learn() 每次的 |TD| 标准差
    all_curve_data = []
    success_count = 0
    tries_since_last_success = 0
    attempts_list = []
    episode_idx = 0
    t0 = time.time()
    # 累计采样量（每次存入replay buffer计数）
    total_samples_collected = 0
    # 额外指标：首次成功、epsilon 达到最小阈值的采样点
    first_success_episode = None
    samples_at_first_success = None
    samples_at_epsilon_min = None
    learning_steps_to_first_success = None
    learn_steps_counter = 0

    # checkpoint 相关
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(output_log)), 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    last_ckpt_time = time.time()
    ckpt_interval_sec = 600

    with open(output_log, 'w') as log_f:
        log_print(log_f, f"==== Start v5 training | batch_size={batch_size}, replay={max_mem_size} | seed={seed} ====")

        while success_count < max_success:
            if max_episodes is not None and episode_idx >= max_episodes:
                log_print(log_f, f"reach max_episodes={max_episodes}, stop.")
                break

            env = ac.ComputerCar(max_vel=400, rotation_vel=4)
            # 记录起点（用于碰撞角度统计）
            start_x, start_y = env.START_POS



            score = 0.0
            done = False
            observation = env.reset_env()
            episode_loss = []
            step_count = 0
            last_reward = 0.0
            ep_start = time.time()

            # 新奖励机制统计（圆心角进度/居中/对齐/无进展）
            ring_cx, ring_cy = ac.WIDTH / 2.0, ac.HEIGHT / 2.0
            prog_thresh = getattr(ac, 'PROGRESS_THRESH', 0.01)
            angle_sum = 0.0
            center_sum = 0.0
            align_sum = 0.0
            np_streak = 0
            np_streak_max = 0
            last_x, last_y = env.x, env.y
            # 直线对齐所需的上一步方向
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
                    td_mean, td_std = agent.get_last_td_stats()
                    if td_mean is not None:
                        td_mean_history.append(td_mean)
                    if td_std is not None:
                        td_std_history.append(td_std)
                    learn_steps_counter += 1
                    # 记录 epsilon 达到最小阈值的采样点
                    if samples_at_epsilon_min is None and getattr(agent, 'epsilon', None) is not None:
                        try:
                            if agent.epsilon <= getattr(agent, 'eps_min', 0.0) + 1e-12:
                                samples_at_epsilon_min = total_samples_collected
                        except Exception:
                            pass
                step_count += 1

                # 基于圆心角的进度与统计
                dx = env.x - last_x
                dy = env.y - last_y
                # 使用数学坐标角度（y 向上）以保证逆时针为正
                theta_prev = math.atan2(ring_cy - last_y, last_x - ring_cx)
                theta_cur = math.atan2(ring_cy - env.y, env.x - ring_cx)
                dtheta = (theta_cur - theta_prev + math.pi) % (2 * math.pi) - math.pi
                angle_progress = max(0.0, dtheta)
                angle_sum += angle_progress

                # 居中因子（与环境一致）
                try:
                    border_dist = env.get_distance_to_border()
                    center_offset = abs(ac.MID_TRACK - border_dist)
                    center_factor = 1.0 - min(1.0, center_offset / max(ac.MID_TRACK, 1e-6))
                except Exception:
                    center_factor = 0.0
                center_sum += center_factor

                # 对齐因子（步进方向与切向方向的一致性）
                step_norm = math.hypot(dx, dy) or 1e-6
                sx, sy = (dx / step_norm, dy / step_norm)
                # 与环境一致的 CCW 切向向量（屏幕坐标）
                tx, ty = -math.sin(theta_cur), -math.cos(theta_cur)
                align_curve = abs(sx * tx + sy * ty)
                if prev_sx is None or prev_sy is None:
                    align_straight = align_curve
                else:
                    align_straight = abs(sx * prev_sx + sy * prev_sy)
                w_curve = min(1.0, abs(dtheta) / max(1e-9, CURVE_DETECT_THRESH))
                align_factor = w_curve * align_curve + (1.0 - w_curve) * align_straight
                align_sum += align_factor

                # 无进展连击（角度进度阈值）
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
            # 更新内存峰值
            current_mem_bytes = _update_mem_peak()
            current_mem_mb = (current_mem_bytes/(1024*1024)) if isinstance(current_mem_bytes,(int,float)) else None

            succeeded = getattr(env, 'is_finished', False) or (last_reward >= success_reward - 1e-6)
            took = time.time() - ep_start

            term_reason = getattr(env, 'termination_reason', 'unknown')

            # 碰撞角度：起点与碰撞点相对圆心的夹角（记最小有符号差值，单位度）
            crash_ang_str = "N/A"
            try:
                if term_reason == 'collision' or getattr(env, 'is_collide', False):
                    # 使用数学坐标角度（y 向上）以保证逆时针为正
                    theta_start = math.atan2(ring_cy - start_y, start_x - ring_cx)
                    theta_crash = math.atan2(ring_cy - env.y, env.x - ring_cx)
                    dtheta = (theta_crash - theta_start + math.pi) % (2 * math.pi) - math.pi
                    crash_ang_str = f"{math.degrees(dtheta):.2f}"
            except Exception:
                crash_ang_str = "N/A"

            # 计算本回合平均统计
            steps_used = max(1, step_count)
            angle_per_step = angle_sum / steps_used
            avg_center = center_sum / steps_used
            avg_align = align_sum / steps_used

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
                # log_print(log_f,
                #     f">>> SUCCESS #{success_count} | ep {episode_idx} | score {score:.2f} | "
                #     f"last_reward {last_reward:.1f} | loss {loss_mean:.3f} | eps {eps_value:.3f} | "
                #     f"steps {step_count} | term {term_reason} | crash_angle_deg {crash_ang_str} | "
                #     f"angle_sum {angle_sum:.4f} | angle/step {angle_per_step:.5f} | "
                #     f"avg_center {avg_center:.3f} | avg_align {avg_align:.3f} | np_max {np_streak_max} | "
                #     f"{tries_note} | {took*1000:.0f} ms"
                # )
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

            now = time.time()
            formatted_time = time.strftime("%Y%m%d%H%M%S", time.localtime(now))
            if now - last_ckpt_time >= ckpt_interval_sec:
                ckpt_path = os.path.join(ckpt_dir, f"ckpt_time_{int(formatted_time)}.pt")
                try:
                    # 再次确保目录存在（长时间训练期间被删除的容错）
                    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                    agent.save_model(ckpt_path)
                    log_print(log_f, f"[ckpt] saved periodic checkpoint -> {ckpt_path}")
                except FileNotFoundError:
                    try:
                        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                        agent.save_model(ckpt_path)
                        log_print(log_f, f"[ckpt] saved periodic checkpoint (retry) -> {ckpt_path}")
                    except Exception as e2:
                        log_print(log_f, f"[ckpt] save failed after retry: {e2}")
                except Exception as e:
                    log_print(log_f, f"[ckpt] save failed: {e}")
                last_ckpt_time = now
            if succeeded:
                ckpt_path = os.path.join(ckpt_dir, f"success_{success_count:03d}_ep_{episode_idx}.pt")
                try:
                    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                    agent.save_model(ckpt_path)
                    log_print(log_f, f"[ckpt] saved success checkpoint -> {ckpt_path}")
                except FileNotFoundError:
                    try:
                        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                        agent.save_model(ckpt_path)
                        log_print(log_f, f"[ckpt] saved success checkpoint (retry) -> {ckpt_path}")
                    except Exception as e2:
                        log_print(log_f, f"[ckpt] save failed after retry: {e2}")
                except Exception as e:
                    log_print(log_f, f"[ckpt] save failed: {e}")

            if (episode_idx % print_every) == 0:
                log_print(log_f,
                    f"ep {episode_idx:5d} | {status:7s} | score {score:9.2f} | "
                    f"last_reward {last_reward:6.1f} | loss {loss_mean:7.3f} | epsilon {eps_value:5.3f} | "
                    f"steps {step_count:4d} | reason: {term_reason} | crash_angle_deg {crash_ang_str} | "
                    f"{tries_note} | {took*1000:.0f} ms"
                )


            # 曲线数据（v4格式）
            all_curve_data.append(f"v5,reward,{episode_idx},{score}\n")
            all_curve_data.append(f"v5,loss,{episode_idx},{loss_mean}\n")
            all_curve_data.append(f"v5,epsilon,{episode_idx},{eps_value}\n")
            if td_mean_history:
                all_curve_data.append(f"v5,td_mean,{episode_idx},{td_mean_history[-1]}\n")
            if td_std_history:
                all_curve_data.append(f"v5,td_std,{episode_idx},{td_std_history[-1]}\n")
            episode_idx += 1

    # success_rate（窗口20）
    success_rate_series = []
    window = 20
    for idx in range(len(success_history)):
        rate = sum(success_history[max(0, idx - window + 1): idx + 1]) / min(idx + 1, window)
        success_rate_series.append(rate)
        all_curve_data.append(f"v5,success_rate,{idx},{rate}\n")

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

    # 缓冲区相关统计
    buffer_capacity = int(getattr(agent.memory, 'mem_size', max_mem_size))
    final_buffer_occupancy = int(min(getattr(agent.memory, 'mem_cntr', total_samples_collected), buffer_capacity))
    samples_dropped = int(max(0, total_samples_collected - buffer_capacity))
    effective_learn_starts = int(max(learn_starts, batch_size))
    avg_samples_per_episode = float(total_samples_collected / episodes) if episodes > 0 else 0.0

    # 附加稳定性指标
    def _var(arr):
        return float(np.var(arr, ddof=0)) if len(arr) > 1 else 0.0
    def _tail(arr, k=100):
        if not arr: return []
        return arr[-k:]
    update_loss_variance = _var(update_loss_history)
    update_loss_variance_last100 = _var(_tail(update_loss_history))
    td_mean_last100 = float(np.mean(_tail(td_mean_history))) if td_mean_history else 0.0
    td_std_last100 = float(np.mean(_tail(td_std_history))) if td_std_history else 0.0

    # 内存结束与峰值（MB）
    mem_end_bytes = None
    try:
        if process:
            mem_end_bytes = process.memory_info().rss
            if mem_peak_bytes is None or (mem_end_bytes and mem_end_bytes > mem_peak_bytes):
                mem_peak_bytes = mem_end_bytes
    except Exception:
        mem_end_bytes = mem_peak_bytes
    def _b2mb(v):
        return (float(v)/(1024*1024)) if isinstance(v,(int,float)) else None
    mem_start_mb = _b2mb(mem_start_bytes)
    mem_end_mb = _b2mb(mem_end_bytes)
    mem_peak_mb = _b2mb(mem_peak_bytes)

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
        f"eps_min_reached_at_samples: {samples_at_epsilon_min if samples_at_epsilon_min is not None else 'N/A'}\n"
        f"buffer_capacity: {buffer_capacity}\n"
        f"final_buffer_occupancy: {final_buffer_occupancy}\n"
        f"samples_dropped_overwritten: {samples_dropped}\n"
        f"effective_learn_starts: {effective_learn_starts}\n"
        f"time_min: {took_total/60:.2f}\n"
        f"episodes_per_min: {epm:.1f}\n"
        f"update_loss_variance: {update_loss_variance:.6f}\n"
        f"update_loss_variance_last100: {update_loss_variance_last100:.6f}\n"
        f"td_mean_last100: {td_mean_last100:.6f}\n"
        f"td_std_last100: {td_std_last100:.6f}\n"
        f"learning_steps_to_first_success: {learning_steps_to_first_success if learning_steps_to_first_success is not None else 'N/A'}\n"
        f"mem_start_mb: {mem_start_mb if mem_start_mb is not None else 'N/A'}\n"
        f"mem_end_mb: {mem_end_mb if mem_end_mb is not None else 'N/A'}\n"
        f"mem_peak_mb: {mem_peak_mb if mem_peak_mb is not None else 'N/A'}\n"
    )

    ensure_parent_dir(output_curve)
    with open(output_curve, 'w') as f:
        f.writelines(all_curve_data)
    ensure_parent_dir(output_summary)
    with open(output_summary, 'w') as f:
        f.write(summary_block)

    print("\n" + summary_block)
    print(f"Curve data saved to {output_curve}")
    print(f"Summary saved to {output_summary}")
    pygame.quit()

    # 返回结果用于 sweep 汇总
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
        'total_samples_collected': int(total_samples_collected),
        'avg_samples_per_episode': float(avg_samples_per_episode),
        'first_success_episode': int(first_success_episode) if first_success_episode is not None else None,
        'samples_at_first_success': int(samples_at_first_success) if samples_at_first_success is not None else None,
        'eps_min_reached_at_samples': int(samples_at_epsilon_min) if samples_at_epsilon_min is not None else None,
        'buffer_capacity': int(buffer_capacity),
        'final_buffer_occupancy': int(final_buffer_occupancy),
        'samples_dropped_overwritten': int(samples_dropped),
        'effective_learn_starts': int(effective_learn_starts),
        'update_loss_variance': update_loss_variance,
        'update_loss_variance_last100': update_loss_variance_last100,
        'td_mean_last100': td_mean_last100,
        'td_std_last100': td_std_last100,
        'learning_steps_to_first_success': int(learning_steps_to_first_success) if learning_steps_to_first_success is not None else None,
        'mem_start_mb': mem_start_mb,
        'mem_end_mb': mem_end_mb,
        'mem_peak_mb': mem_peak_mb,
        # 超参
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


def sweep_worker(task):
    """Worker函数，在子进程中执行单次训练。"""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("FORCE_CPU", "1")
    
    # 准备参数，避免重复传递 batch_size
    run_kwargs = dict(task['run_kwargs'])
    if 'batch_size' in task:
        run_kwargs['batch_size'] = task['batch_size']
    
    res = run_v5_training(
        max_mem_size=task['size'],
        output_curve=task['curve_path'],
        output_summary=task['summary_path'],
        output_log=task['log_path'],
        seed=task['seed'],
        run_size=task['size'],
        run_rep=task['rep'],
        **run_kwargs,
    )
    
    # 计算收敛指标（用于 RQ2 分析）
    episodes_to_convergence = res['first_success_episode'] if res['first_success_episode'] is not None else res['episodes']
    samples_to_convergence = res['samples_at_first_success'] if res['samples_at_first_success'] is not None else res['total_samples_collected']
    
    row = {
        'buffer_size': task['size'],  # 修改为 buffer_size 以匹配分析函数
        'batch_size': task.get('batch_size', task['run_kwargs'].get('batch_size', 128)),
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
        'episodes_per_min': res['episodes_per_min'],
        'total_samples_collected': res['total_samples_collected'],
        'avg_samples_per_episode': res['avg_samples_per_episode'],
        'first_success_episode': res['first_success_episode'],
        'samples_at_first_success': res['samples_at_first_success'],
        'eps_min_reached_at_samples': res['eps_min_reached_at_samples'],
        'buffer_capacity': res['buffer_capacity'],
        'final_buffer_occupancy': res['final_buffer_occupancy'],
        'samples_dropped_overwritten': res['samples_dropped_overwritten'],
        'effective_learn_starts': res['effective_learn_starts'],
        'update_loss_variance': res.get('update_loss_variance'),
        'loss_variance_last100': res.get('update_loss_variance_last100'),  # 修改列名以匹配分析函数
        'td_mean_last100': res.get('td_mean_last100'),
        'td_std_last100': res.get('td_std_last100'),
        'learning_steps_to_first_success': res.get('learning_steps_to_first_success'),
        'mem_start_mb': res.get('mem_start_mb'),
        'mem_end_mb': res.get('mem_end_mb'),
        'mem_peak_mb': res.get('mem_peak_mb'),
        'output_summary': res['output_summary'],
        # RQ2 关键指标
        'episodes_to_convergence': episodes_to_convergence,
        'samples_to_convergence': samples_to_convergence,
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
    """对不同 replay buffer size 进行横向实验并生成 CSV 汇总（支持并行）。"""
    base_dir = Path(base_output_dir)
    if tag:
        base_dir = base_dir / str(tag)
    base_dir.mkdir(parents=True, exist_ok=True)

    csv_path = base_dir / 'sweep_summary.csv'
    fieldnames = [
        'buffer_size', 'batch_size', 'rep', 'episodes', 'successes', 'success_rate', 'avg_score', 'std_score',
        'avg_loss', 'avg_epsilon', 'avg_tries', 'tries_std', 'time_min', 'episodes_per_min',
        'total_samples_collected', 'avg_samples_per_episode', 'first_success_episode', 'samples_at_first_success',
        'eps_min_reached_at_samples', 'buffer_capacity', 'final_buffer_occupancy', 'samples_dropped_overwritten',
        'effective_learn_starts', 'loss_variance_last100', 'episodes_to_convergence', 'samples_to_convergence',
        'output_summary'
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
                'curve_path': curve_path,
                'summary_path': summary_path,
                'log_path': log_path,
                'seed': seed,
                'run_kwargs': run_kwargs,
            }
            tasks.append(task)

    # 自动检测并行度
    def _auto_workers():
        for k in ('SWEEP_WORKERS', 'SLURM_CPUS_PER_TASK'):
            v = os.getenv(k)
            if v and v.isdigit() and int(v) > 0:
                return int(v)
        return max(1, (os.cpu_count() or 1))

    max_workers = _auto_workers()
    print(f"[Sweep] total tasks={len(tasks)}, parallel workers={max_workers}")

    # 并行执行
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

    # 写入CSV
    results.sort(key=lambda r: (r['buffer_size'], r.get('batch_size', 0), r['rep']))
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Sweep finished. Summary CSV: {csv_path}")
    return str(csv_path)


if __name__ == '__main__':
    # 如果同时配置了 BUFFER_SIZES 与 BATCH_SIZES，则进行二维网格 sweep
    def sweep_replay_grid(
        buffer_sizes,
        batch_sizes,
        repeats=1,
        base_output_dir='./v5_faster/v5_exp_data/replay_grid',
        tag=None,
        shared_seed=123,
        **shared_kwargs,
    ):
        base_dir = Path(base_output_dir)
        if tag:
            base_dir = base_dir / str(tag)
        base_dir.mkdir(parents=True, exist_ok=True)

        csv_path = base_dir / 'grid_summary.csv'
        fieldnames = [
            'buffer_size', 'batch_size', 'rep',
            'episodes', 'successes', 'success_rate',
            'avg_score', 'std_score', 'avg_loss', 'avg_epsilon', 'avg_tries', 'tries_std',
            'time_min', 'episodes_per_min',
            'total_samples_collected', 'avg_samples_per_episode',
            'first_success_episode', 'samples_at_first_success', 'eps_min_reached_at_samples',
            'buffer_capacity', 'final_buffer_occupancy', 'samples_dropped_overwritten', 'effective_learn_starts',
            'update_loss_variance', 'loss_variance_last100', 'td_mean_last100', 'td_std_last100',
            'learning_steps_to_first_success', 'mem_start_mb', 'mem_end_mb', 'mem_peak_mb',
            'episodes_to_convergence', 'samples_to_convergence',
            'output_summary'
        ]

        default_ls = int(shared_kwargs.get('learn_starts', 5000))

        # 构建任务列表
        tasks = []
        for size in buffer_sizes:
            for bs in batch_sizes:
                    if int(bs) > int(size):
                        print(f"[Skip] size={size} < batch={bs}, 跳过该组合")
                        continue

                    # 自适应 learn_starts：不超过 0.1*size，且不小于 batch
                    ls = max(int(bs), min(default_ls, int(0.1 * int(size))))

                    for rep in range(1, repeats + 1):
                        out_dir = base_dir / f'size_{int(size)}' / f'batch_{int(bs)}' / f'rep_{rep}'
                        out_dir.mkdir(parents=True, exist_ok=True)

                        curve_path = str(out_dir / 'curve_data.txt')
                        summary_path = str(out_dir / 'summary.txt')
                        log_path = str(out_dir / 'train_log.txt')

                        run_kwargs = dict(shared_kwargs)
                        run_kwargs['batch_size'] = int(bs)
                        run_kwargs['learn_starts'] = int(ls)

                        seed = shared_seed + rep - 1
                        task = {
                            'size': int(size),
                            'batch_size': int(bs),
                            'rep': rep,
                            'curve_path': curve_path,
                            'summary_path': summary_path,
                            'log_path': log_path,
                            'seed': seed,
                            'run_kwargs': run_kwargs,
                        }
                        tasks.append(task)

        # 自动检测并行度
        def _auto_workers():
            for k in ('SWEEP_WORKERS', 'SLURM_CPUS_PER_TASK'):
                v = os.getenv(k)
                if v and v.isdigit() and int(v) > 0:
                    return int(v)
            return max(1, (os.cpu_count() or 1))

        max_workers = _auto_workers()
        print(f"[Grid] total tasks={len(tasks)}, parallel workers={max_workers}")

        # 并行执行
        results = []
        if max_workers <= 1:
            for t in tasks:
                print(f"[Grid] run serial: size={t['size']} batch={t['batch_size']} rep={t['rep']}")
                results.append(sweep_worker(t))
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                fut2task = {ex.submit(sweep_worker, t): t for t in tasks}
                for fut in as_completed(fut2task):
                    t = fut2task[fut]
                    try:
                        row = fut.result()
                        print(f"[Grid] done: size={t['size']} batch={t['batch_size']} rep={t['rep']}")
                        results.append(row)
                    except Exception as e:
                        print(f"[Grid] failed: size={t['size']} batch={t['batch_size']} rep={t['rep']} err={e}")

        # 写入CSV
        results.sort(key=lambda r: (r['buffer_size'], r.get('batch_size', 0), r['rep']))
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for row in results:
                writer.writerow(row)

        print(f"Grid sweep finished. Summary CSV: {csv_path}")
        return str(csv_path)

    # ==========================
    # RQ2 分析函数
    # ==========================
    
    def analyze_rq2_robustness(csv_path: str, output_dir: str):
        """
        RQ2: Batch Size Sensitivity Analysis
        
        Analyzes how batch size moderates the effects of replay buffer size on DQN performance.
        
        Sub-questions:
        - SRQ2.1: Performance trends consistency across batch sizes
        - SRQ2.2: Interaction effects (batch size × buffer size)
        """
        import pandas as pd
        import scipy.stats as stats
        from scipy.stats import spearmanr, pearsonr
        
        print("\n" + "="*80)
        print("RQ2: ROBUSTNESS ANALYSIS - BATCH SIZE SENSITIVITY")
        print("="*80)
        
        # Load data
        df = pd.read_csv(csv_path)
        print(f"\n[Data] Loaded {len(df)} runs from {csv_path}")
        print(f"[Data] Buffer sizes: {sorted(df['buffer_size'].unique())}")
        print(f"[Data] Batch sizes: {sorted(df['batch_size'].unique())}")
        print(f"[Data] Repeats per config: {df.groupby(['buffer_size', 'batch_size']).size().min()}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Key metrics for analysis
        metrics = ['success_rate', 'episodes_to_convergence', 'samples_to_convergence', 
                   'avg_score', 'loss_variance_last100']
        
        results = {}
        
        # ===== SRQ2.1: Performance Trends Consistency =====
        print("\n" + "-"*80)
        print("SRQ2.1: Performance Trends Consistency Across Batch Sizes")
        print("-"*80)
        
        srq21_results = {}
        
        for metric in metrics:
            print(f"\n[Metric: {metric}]")
            
            # Compute mean performance for each (buffer_size, batch_size) combination
            pivot = df.groupby(['buffer_size', 'batch_size'])[metric].mean().unstack(fill_value=np.nan)
            
            # Rank correlation across batch sizes
            buffer_sizes = sorted(df['buffer_size'].unique())
            batch_sizes = sorted(df['batch_size'].unique())
            
            # Compute ranking for each batch size
            rankings = {}
            for bs in batch_sizes:
                if bs in pivot.columns:
                    rankings[bs] = pivot[bs].rank(method='average', ascending=False)
            
            # Pairwise rank correlations
            rank_correlations = []
            for i, bs1 in enumerate(batch_sizes[:-1]):
                for bs2 in batch_sizes[i+1:]:
                    if bs1 in rankings and bs2 in rankings:
                        valid_mask = ~(rankings[bs1].isna() | rankings[bs2].isna())
                        if valid_mask.sum() > 2:
                            rho, p_val = spearmanr(rankings[bs1][valid_mask], rankings[bs2][valid_mask])
                            rank_correlations.append({
                                'batch1': bs1, 'batch2': bs2,
                                'spearman_rho': rho, 'p_value': p_val
                            })
                            print(f"  Rank correlation (batch={bs1} vs {bs2}): ρ={rho:.3f}, p={p_val:.4f}")
            
            mean_rho = np.mean([rc['spearman_rho'] for rc in rank_correlations]) if rank_correlations else np.nan
            print(f"  → Mean Spearman ρ across batch pairs: {mean_rho:.3f}")
            
            srq21_results[metric] = {
                'pivot_table': pivot,
                'rankings': rankings,
                'rank_correlations': rank_correlations,
                'mean_spearman_rho': mean_rho
            }
        
        results['SRQ2.1'] = srq21_results
        
        # ===== SRQ2.2: Interaction Effects (Two-Way ANOVA) =====
        print("\n" + "-"*80)
        print("SRQ2.2: Interaction Effects (Batch Size × Buffer Size)")
        print("-"*80)
        
        srq22_results = {}
        
        try:
            from scipy.stats import f_oneway
            
            for metric in metrics:
                print(f"\n[Metric: {metric}]")
                
                # Prepare data for ANOVA
                df_metric = df[['buffer_size', 'batch_size', metric]].dropna()
                
                if len(df_metric) < 10:
                    print(f"  ⚠️  Insufficient data for {metric}")
                    continue
                
                # Compute group means
                group_means = df_metric.groupby(['buffer_size', 'batch_size'])[metric].mean()
                
                # Overall mean
                grand_mean = df_metric[metric].mean()
                n_total = len(df_metric)
                
                # SS Total
                ss_total = ((df_metric[metric] - grand_mean) ** 2).sum()
                
                # SS Buffer (main effect)
                buffer_means = df_metric.groupby('buffer_size')[metric].mean()
                buffer_counts = df_metric.groupby('buffer_size').size()
                ss_buffer = sum(buffer_counts * (buffer_means - grand_mean) ** 2)
                
                # SS Batch (main effect)
                batch_means = df_metric.groupby('batch_size')[metric].mean()
                batch_counts = df_metric.groupby('batch_size').size()
                ss_batch = sum(batch_counts * (batch_means - grand_mean) ** 2)
                
                # SS Interaction
                ss_cells = sum(
                    df_metric.groupby(['buffer_size', 'batch_size']).size() * 
                    (group_means - grand_mean) ** 2
                )
                ss_interaction = ss_cells - ss_buffer - ss_batch
                
                # SS Error
                ss_error = ss_total - ss_cells
                
                # Degrees of freedom
                n_buffers = df_metric['buffer_size'].nunique()
                n_batches = df_metric['batch_size'].nunique()
                df_buffer = n_buffers - 1
                df_batch = n_batches - 1
                df_interaction = df_buffer * df_batch
                df_error = n_total - n_buffers * n_batches
                df_total = n_total - 1
                
                # Mean squares
                ms_buffer = ss_buffer / df_buffer if df_buffer > 0 else 0
                ms_batch = ss_batch / df_batch if df_batch > 0 else 0
                ms_interaction = ss_interaction / df_interaction if df_interaction > 0 else 0
                ms_error = ss_error / df_error if df_error > 0 else 0
                
                # F-statistics
                f_buffer = ms_buffer / ms_error if ms_error > 0 else np.nan
                f_batch = ms_batch / ms_error if ms_error > 0 else np.nan
                f_interaction = ms_interaction / ms_error if ms_error > 0 else np.nan
                
                # Effect sizes (η²)
                eta2_buffer = ss_buffer / ss_total if ss_total > 0 else 0
                eta2_batch = ss_batch / ss_total if ss_total > 0 else 0
                eta2_interaction = ss_interaction / ss_total if ss_total > 0 else 0
                
                # Partial η²
                partial_eta2_buffer = ss_buffer / (ss_buffer + ss_error) if (ss_buffer + ss_error) > 0 else 0
                partial_eta2_batch = ss_batch / (ss_batch + ss_error) if (ss_batch + ss_error) > 0 else 0
                partial_eta2_interaction = ss_interaction / (ss_interaction + ss_error) if (ss_interaction + ss_error) > 0 else 0
                
                # P-values (approximate using F-distribution)
                from scipy.stats import f as f_dist
                p_buffer = 1 - f_dist.cdf(f_buffer, df_buffer, df_error) if not np.isnan(f_buffer) else np.nan
                p_batch = 1 - f_dist.cdf(f_batch, df_batch, df_error) if not np.isnan(f_batch) else np.nan
                p_interaction = 1 - f_dist.cdf(f_interaction, df_interaction, df_error) if not np.isnan(f_interaction) else np.nan
                
                print(f"\n  Two-Way ANOVA Results:")
                print(f"  {'Source':<20} {'SS':>12} {'df':>6} {'MS':>12} {'F':>10} {'p':>10} {'η²':>10} {'partial η²':>12}")
                print(f"  {'-'*20} {'-'*12} {'-'*6} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
                print(f"  {'Buffer Size':<20} {ss_buffer:>12.2f} {df_buffer:>6} {ms_buffer:>12.2f} {f_buffer:>10.2f} {p_buffer:>10.4f} {eta2_buffer:>10.4f} {partial_eta2_buffer:>12.4f}")
                print(f"  {'Batch Size':<20} {ss_batch:>12.2f} {df_batch:>6} {ms_batch:>12.2f} {f_batch:>10.2f} {p_batch:>10.4f} {eta2_batch:>10.4f} {partial_eta2_batch:>12.4f}")
                print(f"  {'Interaction':<20} {ss_interaction:>12.2f} {df_interaction:>6} {ms_interaction:>12.2f} {f_interaction:>10.2f} {p_interaction:>10.4f} {eta2_interaction:>10.4f} {partial_eta2_interaction:>12.4f}")
                print(f"  {'Error':<20} {ss_error:>12.2f} {df_error:>6} {ms_error:>12.2f}")
                print(f"  {'Total':<20} {ss_total:>12.2f} {df_total:>6}")
                
                # Interpretation
                print(f"\n  Interpretation:")
                sig_level = "***" if p_interaction < 0.001 else "**" if p_interaction < 0.01 else "*" if p_interaction < 0.05 else "n.s."
                effect_size = "large" if eta2_interaction >= 0.14 else "medium" if eta2_interaction >= 0.06 else "small"
                print(f"  - Interaction effect: η²={eta2_interaction:.4f} ({effect_size}), p={p_interaction:.4f} {sig_level}")
                print(f"  - Buffer main effect: η²={eta2_buffer:.4f}, p={p_buffer:.4f}")
                print(f"  - Batch main effect: η²={eta2_batch:.4f}, p={p_batch:.4f}")
                
                if eta2_interaction < 0.1:
                    print(f"  → H2.2 SUPPORTED: Weak interaction (η²={eta2_interaction:.4f} < 0.1)")
                else:
                    print(f"  → H2.2 REJECTED: Strong interaction (η²={eta2_interaction:.4f} ≥ 0.1)")
                
                srq22_results[metric] = {
                    'ss_buffer': ss_buffer, 'df_buffer': df_buffer, 'ms_buffer': ms_buffer, 'f_buffer': f_buffer, 'p_buffer': p_buffer,
                    'ss_batch': ss_batch, 'df_batch': df_batch, 'ms_batch': ms_batch, 'f_batch': f_batch, 'p_batch': p_batch,
                    'ss_interaction': ss_interaction, 'df_interaction': df_interaction, 'ms_interaction': ms_interaction, 
                    'f_interaction': f_interaction, 'p_interaction': p_interaction,
                    'ss_error': ss_error, 'df_error': df_error, 'ms_error': ms_error,
                    'ss_total': ss_total, 'df_total': df_total,
                    'eta2_buffer': eta2_buffer, 'eta2_batch': eta2_batch, 'eta2_interaction': eta2_interaction,
                    'partial_eta2_buffer': partial_eta2_buffer, 'partial_eta2_batch': partial_eta2_batch, 
                    'partial_eta2_interaction': partial_eta2_interaction,
                }
        
        except Exception as e:
            print(f"  ⚠️  ANOVA computation failed: {e}")
        
        results['SRQ2.2'] = srq22_results
        
        # ===== Summary Report =====
        print("\n" + "="*80)
        print("RQ2 SUMMARY")
        print("="*80)
        
        print("\n[SRQ2.1: Consistency Check]")
        for metric in metrics:
            if metric in srq21_results:
                mean_rho = srq21_results[metric]['mean_spearman_rho']
                if not np.isnan(mean_rho):
                    consistency = "HIGH" if mean_rho > 0.7 else "MODERATE" if mean_rho > 0.5 else "LOW"
                    print(f"  {metric:30s}: Mean ρ={mean_rho:.3f} ({consistency} consistency)")
                    if mean_rho > 0.7:
                        print(f"    → H2.1 SUPPORTED: Rankings preserved across batch sizes")
                    else:
                        print(f"    → H2.1 WEAKENED: Some rank changes observed")
        
        print("\n[SRQ2.2: Interaction Strength]")
        interaction_small_count = 0
        interaction_total_count = 0
        for metric in metrics:
            if metric in srq22_results:
                eta2_int = srq22_results[metric]['eta2_interaction']
                p_int = srq22_results[metric]['p_interaction']
                sig = "***" if p_int < 0.001 else "**" if p_int < 0.01 else "*" if p_int < 0.05 else "n.s."
                print(f"  {metric:30s}: η²={eta2_int:.4f} {sig}")
                
                interaction_total_count += 1
                if eta2_int < 0.1:
                    interaction_small_count += 1
        
        if interaction_total_count > 0:
            support_rate = 100 * interaction_small_count / interaction_total_count
            print(f"\n  Overall: {interaction_small_count}/{interaction_total_count} metrics show weak interaction")
            print(f"  → H2.2 Support Rate: {support_rate:.1f}%")
            if support_rate >= 80:
                print(f"  → STRONG SUPPORT for H2.2: Effects are largely independent")
            elif support_rate >= 50:
                print(f"  → MODERATE SUPPORT for H2.2: Mixed interaction patterns")
            else:
                print(f"  → WEAK SUPPORT for H2.2: Significant interactions present")
        
        # Save results
        results_path = os.path.join(output_dir, 'rq2_analysis_results.csv')
        with open(results_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Mean_Spearman_Rho', 'Eta2_Buffer', 'Eta2_Batch', 
                           'Eta2_Interaction', 'P_Interaction', 'H2.1_Supported', 'H2.2_Supported'])
            for metric in metrics:
                rho = srq21_results.get(metric, {}).get('mean_spearman_rho', np.nan)
                eta2_buf = srq22_results.get(metric, {}).get('eta2_buffer', np.nan)
                eta2_bat = srq22_results.get(metric, {}).get('eta2_batch', np.nan)
                eta2_int = srq22_results.get(metric, {}).get('eta2_interaction', np.nan)
                p_int = srq22_results.get(metric, {}).get('p_interaction', np.nan)
                h21 = 'YES' if not np.isnan(rho) and rho > 0.7 else 'NO'
                h22 = 'YES' if not np.isnan(eta2_int) and eta2_int < 0.1 else 'NO'
                writer.writerow([metric, f'{rho:.4f}', f'{eta2_buf:.4f}', f'{eta2_bat:.4f}',
                               f'{eta2_int:.4f}', f'{p_int:.4f}', h21, h22])
        
        print(f"\n[Output] Results saved to: {results_path}")
        print("="*80 + "\n")
        
        return results

    # 主入口
    if BUFFER_SIZES and BATCH_SIZES:
        print(f"[Config] Grid sizes={BUFFER_SIZES}, batches={BATCH_SIZES}, repeats={REPEATS}, tag={TAG}, seed={SHARED_SEED}")
        csv_path = sweep_replay_grid(
            buffer_sizes=BUFFER_SIZES,
            batch_sizes=BATCH_SIZES,
            repeats=REPEATS,
            base_output_dir='./v5_faster/v5_exp_data/replay_grid',
            tag=TAG,
            shared_seed=SHARED_SEED,
            **TRAIN_KWARGS,
        )
        
        # Run RQ2 analysis after sweep completes
        if csv_path and os.path.exists(csv_path):
            tag_dir = os.path.dirname(csv_path)
            output_dir = os.path.join(tag_dir, 'rq2_analysis')
            try:
                analyze_rq2_robustness(csv_path, output_dir)
            except Exception as e:
                print(f"[ERROR] RQ2 analysis failed: {e}")
                import traceback
                traceback.print_exc()
        
    elif BUFFER_SIZES:
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
        run_v5_training(
            **TRAIN_KWARGS,
        )

# ==========================================================================================
# RQ2 数据收集与分析说明
# ==========================================================================================
#
# 【实验设计】
# 1. 缓冲区大小: [100000, 50000, 10000, 5000, 1000] (5 水平)
# 2. 批量大小:   [512, 128, 64, 32, 4] (5 水平)
# 3. 重复次数:   建议 ≥ 3 (当前配置为 3)
# 4. 总运行次数: 5 × 5 × 3 = 75 次
#
# 【关键指标说明】
# - success_rate: 成功率，衡量整体性能稳定性
# - episodes_to_convergence: 首次成功的回合数，衡量收敛速度
# - samples_to_convergence: 首次成功的样本数，衡量样本效率
# - avg_score: 平均得分，衡量平均性能水平
# - loss_variance_last100: 训练后期损失方差，衡量学习稳定性
#
# 【SRQ2.1 分析流程】
# 1. 对每个批量大小，计算各缓冲区的平均性能
# 2. 对每个批量大小，将缓冲区按性能排名
# 3. 计算不同批量大小间的 Spearman 秩相关系数 ρ
# 4. 判断: 若平均 ρ > 0.7，则 H2.1 支持（排名一致）
#
# 【SRQ2.2 分析流程】
# 1. 执行双因素 ANOVA: performance ~ buffer_size + batch_size + buffer×batch
# 2. 计算交互项的效应大小 η² = SS_interaction / SS_total
# 3. 判断: 若 η² < 0.1，则 H2.2 支持（弱交互，效应独立）
#
# 【预期结果解释】
# 情景 1: H2.1 和 H2.2 都支持
#   → 缓冲区和批量大小的效应是独立的，可分别调优
#   → 例如: 大缓冲区总是更好，无论批量大小如何
#
# 情景 2: H2.1 支持但 H2.2 不支持
#   → 排名一致但效应强度随批量大小变化
#   → 例如: 大缓冲区总是最好，但在大批量下优势更明显
#
# 情景 3: H2.1 和 H2.2 都不支持
#   → 存在强交互，最优缓冲区依赖于批量大小
#   → 例如: 小批量需要大缓冲区，大批量需要小缓冲区
#
# 【输出文件】
# - grid_summary.csv: 所有运行的原始数据
# - rq2_analysis/rq2_analysis_results.csv: 分析结果汇总
#   包含每个指标的 Spearman ρ、ANOVA 统计量（η²、p值）、假设支持状态
#
# 【使用建议】
# 1. 快速测试: BUFFER_SIZES=[10000,5000], BATCH_SIZES=[128,32], REPEATS=2
# 2. 完整实验: 使用当前配置，约需 60-125 小时（使用并行）
# 3. 数据验证: 运行完成后检查 CSV 完整性和缺失值
# 4. 结果分析: 自动生成的 rq2_analysis_results.csv 包含所有统计结果
# ==========================================================================================
