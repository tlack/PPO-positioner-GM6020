opt_ranges = {
    "learning_rate": [3e-5],
    "steps_per_episode": 5,
    "action_muting": 0.1,
    "entropy_coef": 0.01,
    "n_episode": 10_0000_000,
    "n_update": 1000 * 10,
    "minibatch": 1000,
    "fixed_seed": 777,
    "PPO_epochs": 20
}

import datetime
import json
import random

import gm6020
import gym_env_gm6020
import ppo_positioner

def make_config(ranges): 
    return {k:(random.choice(v) if type(v) == type([]) else v) for k,v in ranges.items()} 

def train_with_config_motor(cfg, motor):
    reward = ppo_positioner.main(motor, **cfg)
    reward["date"] = datetime.datetime.today()
    reward["config"] = cfg
    j = {k:repr(v) for k,v in reward.items()}
    open("train-log.jsonlines", "a").write(json.dumps(j)+"\n")
    return reward

def train_with_ranges(opt_ranges):
    motor = gm6020.Motor()
    motor.start()
    cfg = make_config(opt_ranges)
    print(f"!!** TRAINING NEW CONFIG:\n\n{repr(cfg)}\n\n")
    train_with_config_motor(cfg, motor)

if __name__ == "__main__":
    while 1:
        train_with_ranges(opt_ranges)

main()
