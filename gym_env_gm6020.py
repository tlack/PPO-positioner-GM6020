ENV_NAME = "gmpos-v1"
MAX_EPISODE_LEN = 10
DIST_THRESH = 5
STEP_SLEEP = 0.1
RESET_SLEEP = 0.75
STEPS_ON_POSITION_FOR_GOAL = 1
ECHO_FREQ = 1

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import os
import math
import numpy as np
import random
import time

# https://stackoverflow.com/a/39662359
def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        return True
    except NameError:
        return False  # Probably standard Python interpreter

if is_notebook():
    from IPython.display import display

class GM6020PositionerEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        self.step_counter = 0
        self.episode_counter = 0
        self.episode_reward = 0
        self.action_muting = 0.2
        self.steps_per_episode = MAX_EPISODE_LEN
        self.action_space = spaces.Box(np.array([-1] * 1), np.array([1] * 1))
        #self.observation_space = spaces.Box(
        #    np.array([0,    -8192,   0,   0, -32767,     0]), 
        #    np.array([8192, 8192, 8192, 360,  32767,   200]))
        self.observation_space = spaces.Box(
            np.array([0,    -8192,   0]),
            np.array([8192, 8192, 8192])
        )
        self.n_goals = 0
        self.last_dx = 0

    def step(self, action):
        done = False

        dv = self.action_muting
        da = action[0]
        if da > 1: da = 1
        if da < -1: da = -1
        
        dx = (0.5*self.last_dx) + (da*dv)
        self.last_dx = dx

        before = self.motor.state
        self.motor.speed(dx)
        time.sleep(STEP_SLEEP)
        self.state = state = self.motor.state
        dist = abs(self.goal - state["pos"])

        reward = 1 - ( (dist*3) / 8191)

        if dist < DIST_THRESH:
            print('!!! ON POINT !!!')
            reward = 1.5
            self.on_position_count += 1
            if self.on_position_count == STEPS_ON_POSITION_FOR_GOAL:
                done = True
                reward = 5
                self.n_goals += 1
        else:
            self.on_position_count = 0

        self.episode_reward += reward
        self.step_counter += 1

        if self.step_counter % ECHO_FREQ == 0:
            line = f"d{dist}\tr{reward:.3f}\tdx{dx:.3f}\tda{da:.3f}\tg{self.goal}\tp{state['pos']}"
            open('log.txt','a').write(line+"\n")
            print(line)

        if self.step_counter > self.steps_per_episode:
            if self.episode_counter % 10 == 0:
                def f(n):
                    return ",".join([f"{x:02f}" for x in n])
                print(
                    f"reward (this, avg/step): {f([reward, self.episode_reward / (self.step_counter+1)])}"
                )
                print(
                    f"goals: {self.n_goals} / {f([self.n_goals / (self.episode_counter+1)])}"
                )
            reward = 0
            done = True

        info = state
        # self.observation = (self.goal, self.goal - state["pos"]) + tuple(state.values())
        self.observation = (self.goal, self.goal - state["pos"], state["pos"])

        if done:
            self.episode_counter += 1

        return np.array(self.observation).astype(np.float32), reward, done, info

    def reset(self):
        self.episode_reward = 0
        self.step_counter = 0
        self.goal = np.floor(random.uniform(0, 8191));
        print('new goal: ',self.goal)
        state = self.motor.state
        # self.observation = (self.goal, self.goal - state["pos"]) + tuple(state.values())
        self.observation = (self.goal, self.goal - state["pos"], state["pos"])
        self.on_position_count = 0
        self.last_dx = 0
        time.sleep(RESET_SLEEP)
        return np.array(self.observation).astype(np.float32)

    def render(self, mode="human"):
        return

    def _get_state(self):
        return self.observation

    def close(self):
        return

gym.register(id=ENV_NAME, entry_point='gym_env_gm6020:GM6020PositionerEnv')

