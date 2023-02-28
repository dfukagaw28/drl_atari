import torch
from torch import nn
import random

class CNNQNetwork(nn.Module):
    def __init__(self,
                state_shape, # (k, c, h, w)
                n_action):
        super(CNNQNetwork, self).__init__()
        k, c, h, w = state_shape
        self.n_action = n_action
        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels=c, out_channels=32,
                    kernel_size=4, stride=2, padding=1, bias=False),
            # (b, c, k, h, w) -> (b, 32, k/2, h/2, w/2)
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=64,
                    kernel_size=4, stride=2, padding=1, bias=False),
            # (b, 32, k/2, h/2, w/2) -> (b, 64, k/4, h/4, w/4)
            nn.ReLU()
        )
        self.fc_in = k*h*w
        self.fc_state = nn.Sequential(
            nn.Linear(self.fc_in, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.fc_advantage = nn.Sequential(
            nn.Linear(self.fc_in, 128),
            nn.ReLU(),
            nn.Linear(128, n_action)
        )

    def forward(self, obs): # (b,k,c,h,w)
        obs = obs.permute((0,2,1,3,4)) # (b,c,k,h,w)
        feature = self.conv_layers(obs)
        feature = feature.contiguous().view(-1, self.fc_in)
        state_values = self.fc_state(feature)
        advantage = self.fc_advantage(feature).view(-1,
                                                self.n_action)
        action_values = state_values + advantage - \
                        torch.mean(advantage, dim=1, keepdim=True)
        return action_values # (b, n_action)

    def act(self, obs, epsilon=0): # actの際はb=1を想定
        if random.random() < epsilon:
            action = random.randrange(self.n_action)
        else:
            with torch.no_grad():
                obs = obs.unsqueeze(0)
                action = torch.argmax(self.forward(obs)).item()
        return action
