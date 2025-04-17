import random
from collections import deque, namedtuple
import numpy as np

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity=100000, n_step=3, gamma=0.99):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)

    def push(self, state, action, reward, next_state, done):
        self.n_step_buffer.append(Transition(state, action, reward, next_state, done))

        if len(self.n_step_buffer) < self.n_step:
            return

        # Compute n-step return
        reward_sum = 0
        for i, transition in enumerate(self.n_step_buffer):
            reward_sum += (self.gamma ** i) * transition.reward

        state_n = self.n_step_buffer[0].state
        action_n = self.n_step_buffer[0].action
        next_state_n = self.n_step_buffer[-1].next_state
        done_n = self.n_step_buffer[-1].done

        self.buffer.append((state_n, action_n, reward_sum, next_state_n, done_n))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
