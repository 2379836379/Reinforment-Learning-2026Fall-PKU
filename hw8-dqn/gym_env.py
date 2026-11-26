from env import Env
import gymnasium as gym

class GymEnv(Env):

    def __init__(self, name):
        self.env = gym.make(name)
        self.state_dim = self.env.observation_space.shape
        self.action_dim = self.env.action_space.n

    def reset(self, conf = {}):
        obs = self.env.reset(**conf)
        # 旧版gym接口返回单个obs，新版接口返回(obs, info)
        if type(obs) == tuple: obs = obs[0]
        return obs
    
    def step(self, action):
        # 旧版gym接口step返回(obs, reward, done, info)，新版接口返回(obs, reward, done, truncated, info)
        obs, reward, *t = self.env.step(action)
        if len(t) == 2:
            return obs, reward, t[0], t[1]
        else:
            done, truncated, info = t
            return obs, reward, done or truncated, info

from collections import deque
import numpy as np


class BreakoutEnv(Env):
    """Four preprocessed frames in CHW order, stored as uint8 for replay."""

    def __init__(self, max_episode_steps=10000):
        import ale_py
        gym.register_envs(ale_py)
        self.env = gym.make(
            'ALE/Breakout-v5', frameskip=4, repeat_action_probability=0.0,
            max_episode_steps=max_episode_steps)
        self.state_dim = (4, 80, 80)
        self.action_dim = self.env.action_space.n
        self.frames = deque(maxlen=4)
        self.fire_action = self.env.unwrapped.get_action_meanings().index('FIRE')
        self.needs_fire = False
        self.lives = 0

    def preprocess_obs(self, obs):
        # Keep the supplied crop/downsample scheme; normalize inside the CNN.
        return np.mean(obs, axis=2)[35:195:2, ::2].astype(np.uint8)

    def reset(self, conf=None):
        obs, info = self.env.reset(**(conf or {}))
        self.frames.clear()
        frame = self.preprocess_obs(obs)
        self.frames.extend([frame] * 4)
        self.lives = info['lives']
        self.needs_fire = True
        return np.stack(self.frames)

    def step(self, action):
        # Launch at game start and after a lost life, including greedy evaluation.
        executed_action = self.fire_action if self.needs_fire else action
        obs, reward, terminated, truncated, info = self.env.step(executed_action)
        self.needs_fire = info['lives'] < self.lives and not (terminated or truncated)
        self.lives = info['lives']
        self.frames.append(self.preprocess_obs(obs))
        info = dict(info, executed_action=executed_action)
        return np.stack(self.frames), reward, terminated or truncated, info
