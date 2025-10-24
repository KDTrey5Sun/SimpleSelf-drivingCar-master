import pygame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from autocar_v3 import ComputerCar, FPS
from DQN_v1 import Agent
import statistics
from draw_v3 import draw, draw_track, WIDTH, HEIGHT, TRACK_WIDTH, LANE_WIDTH, CENTER_X

def plot_with_buttons(curve_dict, title, ylabel, buffer_sizes):
    fig, ax = plt.subplots()
    lines = {}
    colors = {1: 'r', 4: 'g', 8: 'b'}
    for buf in buffer_sizes:
        curve = curve_dict[buf]
        (line,) = ax.plot(curve, label=f'buffer={buf}', color=colors[buf], visible=False, linewidth=2)
        lines[buf] = line
    ax.set_title(title)
    ax.set_xlabel('Episode')
    ax.set_ylabel(ylabel)
    ax.legend()

    plt.subplots_adjust(bottom=0.2)
    button_axes = []
    buttons = []
    for i, buf in enumerate(buffer_sizes):
        ax_btn = plt.axes([0.2 + i*0.2, 0.05, 0.15, 0.075])
        button = Button(ax_btn, f'buffer={buf}')
        button_axes.append(ax_btn)
        buttons.append(button)

    def make_callback(buf):
        def callback(event):
            for b, l in lines.items():
                l.set_visible(b == buf)
            fig.canvas.draw()
        return callback

    for i, buf in enumerate(buffer_sizes):
        buttons[i].on_clicked(make_callback(buf))

    # 默认显示buffer=1
    lines[buffer_sizes[0]].set_visible(True)
    plt.show()




if __name__ == '__main__':
    pygame.init()
    WIN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("DQN Car v3 - Straight Track")
    clock = pygame.time.Clock()

    buffer_sizes = [1, 4, 8]
    max_success = 50
    results = {}

    # For plotting curves
    reward_curve_dict = {}
    loss_curve_dict = {}
    success_rate_curve_dict = {}

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

        while success_count < max_success:
            env = ComputerCar(max_vel=8, rotation_vel=4)
            env.START_POS = (CENTER_X, HEIGHT - 80)
            env.reset()

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
                scores.append(score)
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

        reward_curve_dict[buffer_size] = reward_history
        loss_curve_dict[buffer_size] = loss_history
        # Calculate sliding window success rate
        window = 20
        success_rate_curve = []
        for idx in range(len(success_history)):
            success_rate_curve.append(
                sum(success_history[max(0, idx-window+1):idx+1]) / min(idx+1, window)
            )
        success_rate_curve_dict[buffer_size] = success_rate_curve

    print("\n==== All experiment results ====")
    for buf, metrics in results.items():
        print(f"Replay buffer: {buf} | Average score: {metrics['avg']:.2f} | Std: {metrics['std']:.2f} | Average tries: {metrics['avg_tries']:.2f} | Tries std: {metrics['std_tries']:.2f}")

    # 分别弹出三个窗口，每个窗口有按钮控制显示哪条曲线


    plot_with_buttons(reward_curve_dict, 'Reward Curve', 'Reward', buffer_sizes)
    plot_with_buttons(loss_curve_dict, 'Loss Curve', 'Loss', buffer_sizes)
    plot_with_buttons(success_rate_curve_dict, 'Success Rate (window=20)', 'Success Rate', buffer_sizes)

    pygame.quit()