import os
import time
import random
import pygame
import numpy as np
import statistics
import math
from DQN import Agent


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
    replace_target=2000,
    success_reward=3000,
    windowless=True,
    render=False,
    output_curve='./v5_faster/v5_exp_data/curve_data_v5.txt',
    output_summary='./v5_faster/v5_exp_data/summary_v5.txt',
    output_log='./v5_faster/v5_exp_data/train_log_v5.txt',
    print_every=1,
    max_episodes=None,
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
        # 不强制 double/dueling，沿用 Agent 默认
        clip_reward=False,
        target_soft_tau=0.0,
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
    ckpt_interval_sec = 600

    with open(output_log, 'w') as log_f:
        log_print(log_f, f"==== Start v5 training | batch_size={batch_size}, replay={max_mem_size} ====")

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
                observation = observation_

                loss = agent.learn()
                if loss is not None:
                    episode_loss.append(loss)
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
                    agent.save_model(ckpt_path)
                    log_print(log_f, f"[ckpt] saved periodic checkpoint -> {ckpt_path}")
                except Exception as e:
                    log_print(log_f, f"[ckpt] save failed: {e}")
                last_ckpt_time = now
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
                    f"last_reward {last_reward:6.1f} | loss {loss_mean:7.3f} | epsilon {eps_value:5.3f} | "
                    f"steps {step_count:4d} | reason: {term_reason} | crash_angle_deg {crash_ang_str} | "
                    f"{tries_note} | {took*1000:.0f} ms"
                )


            # 曲线数据（v4格式）
            all_curve_data.append(f"v5,reward,{episode_idx},{score}\n")
            all_curve_data.append(f"v5,loss,{episode_idx},{loss_mean}\n")
            all_curve_data.append(f"v5,epsilon,{episode_idx},{eps_value}\n")
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

    summary_block = (
        "==== Summary (v5) ====\n"
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

    with open(output_curve, 'w') as f:
        f.writelines(all_curve_data)
    with open(output_summary, 'w') as f:
        f.write(summary_block)

    print("\n" + summary_block)
    print(f"Curve data saved to {output_curve}")
    print(f"Summary saved to {output_summary}")
    pygame.quit()


if __name__ == '__main__':
    run_v5_training(
        max_success=100,
        batch_size=128,
        max_mem_size=100000,
        windowless=True,
        render=False,
    )
