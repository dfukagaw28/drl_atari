from atari_wrappers import *
import argparse
import cv2
import gym
import numpy as np
import pathlib
import torch
import sys
sys.path.append(".")
from models.dueling_dqn import CNNQNetwork

def make_pong_env(noop_max=30, skip=4, width=84, height=84):
    env = gym.make('PongNoFrameskip-v4')
    env = NoopResetEnv(env, noop_max=noop_max)
    env = MaxAndSkipEnv(env, skip=skip)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = WarpFrame(env, width=width, height=height, grayscale=True)
    env = ClipRewardEnv(env)
    env = FrameStack(env, skip)
    return env

def make_pong_video(env, video_path, width=84, height=84, reward_font_size=20, model=None):
    obs = env.reset()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(str(video_path), fourcc, 15, (width, height), isColor=0)
    total_reward = 0
    done = False
    while not done:
        frame = obs[-1][0].detach().cpu().numpy() # (c,h,w)
        #frame = frame.transpose(1,2,0).astype(np.uint8)
        #frame = cv2.putText(frame, f"Reward:{int(total_reward)}", (0, height-reward_font_size),
        #                    cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,0), 1, cv2.LINE_AA)
        video.write(frame)
        if model is not None:
            with torch.no_grad():
                obs = obs.float().to(device)
                action = model.act(obs)
        else:
            action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
    video.release()
    return

if __name__=="__main__":
    root_path = pathlib.Path(".")
    parser = argparse.ArgumentParser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--video_name", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--load_name", type=str)
    args = parser.parse_args()
    env = make_pong_env()
    video_name = args.video_name
    model_name = args.model_name
    load_name = args.load_name
    videos_path = root_path / "videos"
    checkpoints_path = root_path / "checkpoints"
    if model_name == "CNNQNetwork":
        model = CNNQNetwork(env.observation_space.shape,
                            n_action=env.action_space.n).to(device)
        load_path = checkpoints_path / load_name
        checkpoint = torch.load(load_path,
                                map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["net_state_dict"])
    else:
        model = None
    if not videos_path.exists():
        videos_path.mkdir(parents=True)
    video_path = videos_path / video_name
    make_pong_video(env, video_path, model=model)
