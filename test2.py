import matplotlib.pyplot as plt
import numpy as np

episodes = np.arange(1, 201)
# 理想分数曲线：前期波动大，后期收敛
scores = 1000 * (1 - np.exp(-episodes / 60)) + np.random.normal(0, 80, size=episodes.shape)
scores = np.clip(scores, 0, None)  # 分数不为负

plt.figure(figsize=(8, 4))
plt.plot(episodes, scores, label='DQN Score')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('Ideal DQN Training Score Curve')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()