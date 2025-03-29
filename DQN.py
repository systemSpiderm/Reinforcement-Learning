import gym

gym_version = gym.__version__
assert gym_version == "0.23.0", f"Expected gym version 0.23.0, but got {gym_version}"
print("Gym version is correct!")

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from collections import namedtuple
import random
import matplotlib.pyplot as plt

class QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNet, self).__init__()
        # 两个线性层
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.Tensor(np.array(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

    def __len__(self):
        return len(self.buffer)

    def push(self, *transition):
        # append(None)是为了占位，下面可以直接按下标进行访问
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = self.transition(*transition)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        # 在buffer中随机取样，batch_size
        transitions = random.sample(self.buffer, batch_size)
        batch = self.transition(*zip(*transitions))
        return batch.state, batch.action, batch.reward, batch.next_state, batch.done

    def clean(self):
        self.buffer = []
        self.position = 0

class DQN:
    def __init__(self, env, input_size, hidden_size, output_size):
        self.env = env
        self.eval_net = QNet(input_size, hidden_size, output_size)
        self.target_net = QNet(input_size, hidden_size, output_size)
        self.optim = optim.Adam(self.eval_net.parameters(), lr=args.lr)
        self.eps = args.eps
        self.buffer = ReplayBuffer(args.capacity)
        self.loss_fn = nn.MSELoss()
        self.learn_step = 0

    def choose_action(self, obs):
        if np.random.rand() < self.eps:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                return torch.argmax(self.eval_net(obs)).item()

    def store_transition(self, *transition):
        self.buffer.push(*transition)

    def learn(self):
        if len(self.buffer) < args.batch_size:
            return
        # 滞后更新 target_net
        if self.learn_step % args.update_target == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step += 1

        obs, actions, rewards, next_obs, dones = self.buffer.sample(args.batch_size)
        actions = torch.LongTensor(actions).unsqueeze(1)
        dones = torch.FloatTensor(dones)
        rewards = torch.FloatTensor(rewards)
        # 智能体决定采取的动作对应的Q值
        q_eval = self.eval_net(obs).gather(1, actions)
        q_next = self.target_net(next_obs).max(1)[0].detach()
        # 根据是否完成，实现q_target的赋值
        q_target = rewards + args.gamma * (1 - dones) * q_next

        loss = self.loss_fn(q_eval, q_target.unsqueeze(1))
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

def main():
    env = gym.make(args.env)
    o_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n
    agent = DQN(env, o_dim, args.hidden, a_dim)

    scores = []
    avg_scores = []
    for episode in range(args.n_episodes):
        state = env.reset()
        total_reward = 0
        for t in range(500):
            # 采取np库的线性插值获得动态下降的探索率
            agent.eps = np.interp(episode * 500 + t, [0, 100000], [1.0, 0.01])
            action = agent.choose_action(state)
            env.render()
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            if done:
                break
            agent.learn()
        # 记录得分
        scores.append(total_reward)
        avg_reward = np.mean(scores[-100:])
        avg_scores.append(avg_reward)
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Average Reward: {avg_reward}, Reward: {scores[episode]}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.xlabel('Episode')
    plt.ylabel('Single Episode Reward')
    plt.title('Single Episode Reward Curve')
    
    plt.subplot(1, 2, 2)
    plt.plot(avg_scores)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (Last 100 Episodes)')
    plt.title('Average Reward Curve')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",        default="CartPole-v1",  type=str,   help="environment name")
    parser.add_argument("--lr",             default=1e-3,       type=float, help="learning rate")
    parser.add_argument("--hidden",         default=64,         type=int,   help="dimension of hidden layer")
    parser.add_argument("--n_episodes",     default=500,        type=int,   help="number of episodes")
    parser.add_argument("--gamma",          default=0.99,       type=float, help="discount factor")
    # parser.add_argument("--log_freq",       default=100,        type=int)
    parser.add_argument("--capacity",       default=10000,      type=int,   help="capacity of replay buffer")
    parser.add_argument("--eps",            default=1.0,        type=float, help="epsilon of explosion")
    # parser.add_argument("--eps_min",        default=0.05,       type=float)
    parser.add_argument("--batch_size",     default=128,        type=int)
    # parser.add_argument("--eps_decay",      default=0.999,      type=float)
    parser.add_argument("--update_target",  default=100,        type=int,   help="frequency to update target network")
    args = parser.parse_args()
    main()
