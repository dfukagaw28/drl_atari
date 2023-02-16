from atari_wrappers import *
import argparse
import cv2
import gym
import numpy as np
import pathlib
root_path = pathlib.Path(".")
parser = argparse.ArgumentParser()
parser.add_argument("--video_name", type=str)
args = parser.parse_args()

def make_pong_env(noop_max=30, skip=4, width=224, height=224):
    env = gym.make('PongNoFrameskip-v4')
    env = NoopResetEnv(env, noop_max=noop_max)
    env = MaxAndSkipEnv(env, skip=skip)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = WarpFrame(env, width=width, height=height, grayscale=False)
    env = ClipRewardEnv(env)
    env = FrameStack(env, skip)
    return env

def make_pong_video(env, video_path, width=224, height=224, reward_font_size=20, model=None):
    obs = env.reset()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(str(video_path), fourcc, 15.0, (width, height))
    total_reward = 0
    done = False
    while not done:
        #for i in range(len(obs)):
        #    frame = obs[i]
        frame = obs[-1]
        frame = cv2.putText(frame, f"Reward:{int(total_reward)}", (0, height-reward_font_size),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,0), 1, cv2.LINE_AA)
        video.write(frame)
        if model is not None:
            action = model(obs)
        else:
            action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
    video.release()
    return

if __name__=="__main__":
    video_name = args.video_name
    videos_path = root_path / "videos"
    if not videos_path.exists():
        videos_path.mkdir(parents=True)
    video_path = videos_path / video_name
    env = make_pong_env()
    make_pong_video(env, video_path)

