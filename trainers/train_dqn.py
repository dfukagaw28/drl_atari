import argparse
import pathlib
import torch
from torch import nn, optim
import sys
sys.path.append(".")
sys.path.append("./utils")
from models.dueling_dqn import CNNQNetwork
from utils.pong_env import make_pong_env
from utils.replay_buffer import PrioritizedReplayBuffer

def train_dqn(env, net, target_net,
            replay_buffer, gamma, batch_size, n_episodes, initial_buffer_size,
            epsilon_, beta_, target_update_interval,
            device, best_save_path, last_save_path, load_path=None):
    print(f"device: {device}")
    if load_path is not None:
        checkpoint = torch.load(load_path,
                                map_location=torch.device("cpu"))
        net.load_state_dict(checkpoint["net_state_dict"])
        target_net.load_state_dict(checkpoint["net_state_dict"])
        current_episode = checkpoint["current_episode"]
        n_episodes += current_episode
        rewards = checkpoint["rewards"]
        episode_losses = checkpoint["episode_losses"]
        best_reward = checkpoint["best_reward"]
    else:
        current_episode = 0
        rewards = []
        episode_losses = []
        best_reward = -1e+10
    #---
    # 1. initialize net, criterion, optimizer
    #---
    criterion = nn.SmoothL1Loss(reduction='none')
    optimizer = optim.AdamW(net.parameters(), lr=1e-06)
    net.to(device)
    target_net.to(device)
    #---
    # 2. episode
    #---
    for episode in range(current_episode, n_episodes):
        #---
        # a. observe initial state
        #---
        obs = env.reset()
        done = False
        episode_reward = 0
        step = 0
        if len(replay_buffer) < initial_buffer_size:
            epsilon = 1
        else:
            epsilon = max(0.01, epsilon_ ** (episode+1))
        beta = max(0.4, 1 - beta_ ** (episode+1))
        episode_loss = 0
        #---
        # b. loop until the episode ends
        #---
        while not done:
            obs = obs.float().to(device)
            #---
            # i. choose action following epsilon greedy policy
            #---
            action = net.act(obs, epsilon)
            #---
            # ii. act and observe next_obs, reward, done
            #---
            next_obs, reward, done, _ = env.step(action)
            obs = obs.cpu()
            next_obs = next_obs.cpu()
            #---
            # iii. push experience to the replay buffer
            #---
            replay_buffer.push([obs, action, reward, next_obs, done])
            obs = next_obs
            if len(replay_buffer) > initial_buffer_size:
                #---
                # iv. sample experience from the replay buffer
                #---
                s_obs, s_action, s_reward, s_next_obs, s_done, s_indices, s_weights = \
                replay_buffer.sample(batch_size, beta=beta)
                s_obs = s_obs.to(device)
                s_action = s_action.to(device)
                s_reward = s_reward.to(device)
                s_next_obs = s_next_obs.to(device)
                s_done = s_done.to(device)
                s_weights = s_weights.to(device)
                #---
                # vi. update network
                #---
                q_values = net(s_obs).gather(1, s_action.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    t_action = torch.argmax(net(s_next_obs), dim=1)
                    q_values_next = \
                    target_net(s_next_obs).gather(1, t_action.unsqueeze(1)).squeeze(1)
                target_q_values = s_reward + gamma * q_values_next * (1-s_done)
                optimizer.zero_grad()
                loss = (s_weights * criterion(q_values, target_q_values)).sum()
                loss.backward()
                optimizer.step()
                episode_loss += loss.item()
                priorities = (target_q_values - q_values).abs().detach().cpu().numpy()
                replay_buffer.update_priorities(s_indices, priorities)
            step += 1
            episode_reward += reward
        #---
        # c. update target network
        #---
        if (episode + 1) % target_update_interval == 0:
            target_net.load_state_dict(net.state_dict())
        episode_loss /= step
        rewards.append(episode_reward)
        episode_losses.append(episode_loss)
        print(f"episode:{episode+1}/{n_episodes} reward:{int(episode_reward)}" +
            f" step:{step} loss:{episode_loss:.4f}" +
            f" epsilon:{epsilon:.4f} beta:{beta:.4f} last action:{action}")
        if best_reward <= episode_reward:
            best_reward = episode_reward
            save_path = best_save_path
        else:
            save_path = last_save_path
        torch.save(
            {
                "net_state_dict": net.state_dict(),
                "current_episode": episode,
                "rewards": rewards,
                "episode_losses": episode_losses,
                "best_reward": best_reward
            },
            str(save_path)
        )
        print(f"net saved to >> {str(save_path)}")
        print()
    return

if __name__=="__main__":
    root_path = pathlib.Path(".")
    parser = argparse.ArgumentParser()
    parser.add_argument("--best_save_name", type=str)
    parser.add_argument("--last_save_name", type=str)
    parser.add_argument("--load_name", type=str)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    buffer_size = 10000
    initial_buffer_size = 1000
    replay_buffer = PrioritizedReplayBuffer(buffer_size)
    env = make_pong_env()
    net = CNNQNetwork(env.observation_space.shape,
                    n_action=env.action_space.n).to(device)
    target_net = CNNQNetwork(env.observation_space.shape,
                            n_action=env.action_space.n).to(device)
    gamma = 0.99
    batch_size = 32
    n_episodes = 1000
    epsilon_ = 0.95
    beta_ = 0.99
    target_update_interval = 2
    checkpoints_path = root_path / "checkpoints"
    if not checkpoints_path.exists():
        checkpoints_path.mkdir(parents=True)
    best_save_name = args.best_save_name
    best_save_path = checkpoints_path / best_save_name
    last_save_name = args.last_save_name
    last_save_path = checkpoints_path / last_save_name
    load_name = args.load_name
    if load_name is not None:
        load_path = checkpoints_path / load_name
    else:
        load_path = None
    train_dqn(env, net, target_net,
            replay_buffer, gamma, batch_size, n_episodes, initial_buffer_size,
            epsilon_, beta_, target_update_interval,
            device, best_save_path, last_save_path, load_path)
