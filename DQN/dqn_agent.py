import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from collections import deque
import random
import matplotlib.pyplot as plt
import os
from datetime import datetime

from DQN.dqn_network import QNetwork
from DQN.replay_buffer import ReplayBuffer


class DQNagent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, gamma=0.99,
                 lr=1e-3, batch_size=64, buffer_capacity=100000,
                 target_update_freq=100, n_step=3):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_step = n_step

        # Q-networks
        self.q_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity, n_step=n_step, gamma=gamma)
        self.update_count = 0
        self.target_update_freq = target_update_freq

        # Epsilon-greedy
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.reward_log = []

    def select_action(self, state, return_index=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if np.random.rand() < self.epsilon:
            index = np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                q_values = self.q_net(state)
            index = q_values.argmax().item()
        
        if return_index:
            return index
        else:
            # action ( 0~3)， model_index = action
            return index

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute current Q values
        curr_q = self.q_net(states).gather(1, actions)

        # Compute target Q values with N-step returns
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (self.gamma ** self.n_step) * (1 - dones) * next_q

        # Compute loss
        loss = nn.MSELoss()(curr_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.q_net.state_dict())
        print(f"Model loaded from {path}")

    def train(self, env, num_episodes=1000, max_timesteps=1000, save_path=None):
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path)

        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            done = False

            for t in range(max_timesteps):
                #action = self.select_action(state)
                model_index = self.select_action(state, return_index=True)  # use DQN to get model id
                env.set_reference_model(model_index)  # update MRAC reference model

                action = model_index  # if action=reference index, you can map to discrete engine control if needed
                
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                self.store_transition(state, action, reward, next_state, float(done))
                self.update()

                state = next_state
                total_reward += reward

                if done or truncated:
                    break

            self.reward_log.append(total_reward)

            print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Epsilon = {self.epsilon:.3f}")
            if reward == 100:  # 成功着陆时 lunarlander 给 +100 分
                print(f" Success landing in Episode {episode+1}")


        if save_path and (episode + 1) % 100 == 0:
            filename = os.path.join(save_path, f"best.pth")
            self.save(filename)

    def test(self, num_episodes=10, max_timesteps=1000, video_path="./videos/test_run.mp4"):
        import gym
        #import imageio
        import os
        frames = []
        #env = gym.make("LunarLander-v2", render_mode="human")
        from env.DQN_MRACenv import LunarLanderDQNMRACWrapper
        # Ensure video folder exists
        #if video_path:
            #os.makedirs(os.path.dirname(video_path), exist_ok=True)
        env = LunarLanderDQNMRACWrapper(render=True, exo_mode='sin')
        log = {
            "u_left": [],
            "u_main": [],
            "u_right": [],
            "xi": [],
            "x_m": [],
            "x": []
        }

        env.use_mrac = True
        self.epsilon = 0.0  # Pure exploitation

        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            done = False

            for t in range(max_timesteps):
                #action = self.select_action(state)
                # 让 DQN 选择参考模型 index
                model_index = self.select_action(state, return_index=True)
                env.set_reference_model(model_index)  # 将模型注入 MRAC 控制器
                
                action = model_index  # 可用于 reward shaping 或主引擎控制
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                state = next_state
                #if video_path:
                frame = env.render()
                    #if isinstance(frame, np.ndarray) and frame.ndim == 3:
                        #frames.append(frame)
                    #else:
                        #print(f"[WARNING] Skipped an invalid frame at episode {episode+1}, timestep {t}")
                '''
                if "mrac_control" in info:
                    u = info["mrac_control"]
                    x_m = info["x_m"]
                    x = info["x"]
                    xi = info["xi"]

                    log["u_left"].append(u[0])
                    log["u_main"].append(u[1])
                    log["u_right"].append(u[2])
                    log["x_m"].append(x_m)
                    log["x"].append(x)
                    log["xi"].append(xi)
                '''
                if done or truncated:
                    break

            print(f"[TEST] Episode {episode + 1}: Reward = {total_reward:.2f}")
        #if video_path and frames:
            #imageio.mimsave(video_path, frames, fps=30)
            #print(f"📽️ Video saved to: {video_path}")
        env.close()
        #plot_mrac_results(log)
  

    def plot_training_curve(self):
        import matplotlib.pyplot as plt

        rewards = self.reward_log
        episodes = np.arange(len(rewards))

     
        window = 10
        if len(rewards) >= window:
            avg_rewards = np.convolve(rewards, np.ones(window) / window, mode='valid')
        else:
            avg_rewards = rewards

      
        success_flags = [1 if r > 90 else 0 for r in rewards]
        success_rate = np.convolve(success_flags, np.ones(window) / window, mode='valid')

        plt.figure(figsize=(12, 6))

        # 1. 
        plt.subplot(2, 1, 1)
        plt.plot(episodes, rewards, label='Reward per Episode', alpha=0.5)
        plt.plot(np.arange(len(avg_rewards)), avg_rewards, label='u-Controller', linewidth=2)
        plt.title("Reward and u-Controller")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.grid(True)

        # 2. 
        plt.subplot(2, 1, 2)
        plt.plot(np.arange(len(success_rate)), success_rate, label='Success Rate', color='green')
        plt.title("Landing Success Rate (smoothed)")
        plt.xlabel("Episode")
        plt.ylabel("Success Rate (last {} episodes)".format(window))
        plt.ylim(0, 1.05)
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

import matplotlib.pyplot as plt

def plot_mrac_results(log):
    u_l = log["u_left"]
    u_m = log["u_main"]
    u_r = log["u_right"]
    xi = np.array(log["xi"])
    x_m = np.array(log["x_m"]).squeeze()
    x = np.array(log["x"]).squeeze()

    t = np.arange(len(u_l))

    plt.figure(figsize=(15, 12))

    # 控制信号与干扰
    plt.subplot(3, 1, 1)
    plt.plot(t, u_l, label="u_left")
    plt.plot(t, u_m, label="u_main")
    plt.plot(t, u_r, label="u_right")
    plt.plot(t, xi[:, 0], "--", label="xi_1 (disturb x)")
    plt.plot(t, xi[:, 1], "--", label="xi_2 (disturb y)")
    plt.title("MRAC Control Signals vs. Exosystem Disturbance")
    plt.xlabel("Time step")
    plt.ylabel("Control / Disturbance")
    plt.legend()
    plt.grid(True)

    # 参考模型轨迹
    plt.subplot(3, 1, 2)
    plt.plot(x_m[:, 0], x_m[:, 1], label="Reference x_m (x vs y)")
    plt.title("Reference Model Trajectory")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    # 实际轨迹
    plt.subplot(3, 1, 3)
    plt.plot(x[:, 0], x[:, 1], label="Plain Plant x (x vs y)")
    plt.title("Plain Model Trajectory (Actual)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
