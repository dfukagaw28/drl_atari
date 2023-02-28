import numpy as np
import torch

class PrioritizedReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.index = 0
        self.buffer = []
        self.priorities = np.zeros(buffer_size, dtype=np.float32)
        self.priorities[0] = 1.0

    def __len__(self):
        return len(self.buffer)

    def push(self, experience):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.index] = experience
        self.priorities[self.index] = self.priorities.max()
        self.index = (self.index + 1) % self.buffer_size

    def sample(self, batch_size, alpha=0.6, beta=0.4):
        if len(self.buffer) < self.buffer_size:
            priorities = self.priorities[:self.index]
        else:
            priorities = self.priorities[:self.buffer_size]
        prob = (priorities ** alpha) / (priorities ** alpha).sum()
        indices = np.random.choice(len(prob), batch_size, p=prob)
        weights = (1 / (len(indices) * prob[indices])) ** beta
        obs, action, reward, next_obs, done = zip(*[self.buffer[i] for i in indices])
        obs = torch.stack(obs, 0).float()
        action = torch.as_tensor(action)
        reward = torch.as_tensor(reward).float()
        next_obs = torch.stack(next_obs, 0).float()
        done = torch.as_tensor(done, dtype=torch.uint8)
        weights = torch.as_tensor(weights).float()
        return (obs, action, reward, next_obs, done, indices,
                weights)

    def update_priorities(self, indices, priorities):
        self.priorities[indices] = priorities + 1e-04
