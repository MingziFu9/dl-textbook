"""实验 10-1 看 DQN 学会平衡小车（CartPole）。

环境：gymnasium 的 CartPole-v1。小车上立着一根杆子，每一步可以向左或向右推小车；
杆子不倒、小车不出界，每坚持一步得 1 分，最多 500 步。
做法：DQN——用一个小神经网络代替 Q 表，输入 4 个状态数，输出“向左”“向右”两个动作的分数（Q 值）。
运行：python lab10_1_dqn_cartpole.py（在 labs/ch10 目录下，约 1～3 分钟）
"""
import random
import sys
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
from torch import nn

sys.path.append("..")
from dlbook import plot_curves, set_seed  # noqa: E402

set_seed(42)
env = gym.make("CartPole-v1")
env.action_space.seed(42)                # 让“随便试”的动作也可以重复
print("状态：小车位置、小车速度、杆子角度、杆子角速度；动作：0 向左推，1 向右推")


def make_net():
    return nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))


q_net = make_net()                       # 正在学习的网络
target_net = make_net()                  # “目标网络”：定期从 q_net 复制参数，让学习目标更稳定
target_net.load_state_dict(q_net.state_dict())
opt = torch.optim.Adam(q_net.parameters(), lr=0.001)
memory = deque(maxlen=20000)             # 经验回放池：存下玩过的每一步（状态、动作、奖励、下一个状态、是否结束）
GAMMA = 0.99                             # 折扣：未来的奖励打多少折


def choose_action(state, epsilon):
    """ε-贪心：以 ε 的概率随便试，否则选 Q 值最大的动作。"""
    if random.random() < epsilon:
        return env.action_space.sample()
    with torch.no_grad():
        return int(q_net(torch.tensor(state)).argmax())


def learn(batch_size=64):
    batch = random.sample(memory, batch_size)
    s, a, r, s2, done = map(np.array, zip(*batch))
    s, s2 = torch.tensor(s), torch.tensor(s2)
    a, r, done = torch.tensor(a), torch.tensor(r, dtype=torch.float32), torch.tensor(done, dtype=torch.float32)
    q = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)              # 当时所选动作的 Q 值
    with torch.no_grad():
        target = r + GAMMA * target_net(s2).max(1).values * (1 - done)   # 目标：这一步的奖励 + 打折后的未来最好得分
    loss = nn.functional.smooth_l1_loss(q, target)
    opt.zero_grad()
    loss.backward()
    opt.step()


scores, avg = [], []
epsilon, steps, start = 1.0, 0, time.time()
best_avg, best_episode, best_state = -1, 0, None
for episode in range(1, 501):
    state, _ = env.reset(seed=episode)
    total, done = 0, False
    while not done:
        action = choose_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        memory.append((state, action, reward, next_state, terminated))
        state, total, steps = next_state, total + reward, steps + 1
        if len(memory) >= 1000:
            learn()
        if steps % 500 == 0:
            target_net.load_state_dict(q_net.state_dict())
    epsilon = max(0.05, epsilon * 0.98)                          # 越来越少地“随便试”
    scores.append(total)
    avg.append(float(np.mean(scores[-20:])))
    if episode >= 20 and avg[-1] > best_avg:                     # 和第 4 章的早停一样：记住表现最好时的参数
        best_avg, best_episode = avg[-1], episode
        best_state = {k: v.clone() for k, v in q_net.state_dict().items()}
    if episode % 50 == 0:
        print(f"  第 {episode:3d} 局：得分 {total:5.0f}，最近 20 局平均 {avg[-1]:6.1f}，ε = {epsilon:.3f}，"
              f"已用 {time.time() - start:.0f} 秒")

plot_curves({"每局得分": scores, "最近 20 局平均": avg}, "lab10_1_scores.png", "DQN 玩 CartPole 的得分")

# 训练结束后，取“最近 20 局平均”最高时的参数，不再随便试，看 10 局的表现
print(f"\n最近 20 局平均得分最高在第 {best_episode} 局：{best_avg:.1f}，用这时的参数测试")
q_net.load_state_dict(best_state)
test = []
for i in range(10):
    state, _ = env.reset(seed=1000 + i)
    total, done = 0, False
    while not done:
        state, reward, terminated, truncated, _ = env.step(choose_action(state, 0.0))
        total += reward
        done = terminated or truncated
    test.append(total)
print(f"训练后测试 10 局得分：{[int(t) for t in test]}，平均 {np.mean(test):.1f}（满分 500）")
rand = []
for i in range(10):
    env.reset(seed=1000 + i)
    total, done = 0, False
    while not done:
        _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        total += reward
        done = terminated or truncated
    rand.append(total)
print(f"作为对比，随机乱推 10 局：平均 {np.mean(rand):.1f}")
torch.save(q_net.state_dict(), "dqn_cartpole.pt")
