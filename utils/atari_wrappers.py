# ref -> https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py

from collections import deque
import cv2
import gym
import numpy as np
import torch

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        noops = self.env.unwrapped.np_random.integers(1, self.noop_max+1)
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape,
                                    dtype=np.uint8)
        self._skip       = skip

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def reset(self, **kwargs):
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            done = True
        self.lives = lives
        return obs, reward, done, info

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.fire_action = 1
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(self.fire_action)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=224, height=224,
                grayscale=True):
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3
        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        self.observation_space = new_space

    def observation(self, obs):
        if self._grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        obs = cv2.resize(
            obs, (self._width, self._height),
            interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            obs = np.expand_dims(obs, -1)
        return obs

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        return np.sign(reward)

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        h, w, c = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                            shape=(k, c, h, w),
                                            dtype=env.observation_space.dtype)

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.k):
            self.frames.append(obs)
        obs = self._get_obs()
        obs = torch.from_numpy(obs._force().transpose(0,3,1,2))
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        obs = self._get_obs()
        obs = torch.from_numpy(obs._force().transpose(0,3,1,2))
        return obs, reward, done, info

    def _get_obs(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class LazyFrames:
    def __init__(self, frames):
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.stack(self._frames)
            self._frames = None
        return self._out # (k, h, w, c)

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]
