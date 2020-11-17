# RoboMaster-Master

The DJI Robomaster GM6020 brushless motor can report its position faithfully in the range 
0..8191 at a rate of 1,000 times a second, but lacks the ability to rotate to a specific 
position.

Most would use a PID controller to handle that, but I was curious about using reinforcement
learning. As of this point, 40k episodes in, not much luck.

This code uses Linux's built in CAN driver. I used a Raspberry Pi 4 and CAN hat.

## Contents

- test.py - manual test script to exercise GM6020

- train.py - choses hyperparameters randomly from options, dispatches PPO

- ppo_positioner.py - PPO implementation 

- gym_env_gm6020.py - OpenAI Gym that attempts to locate randomly chosen goal position on motor

- gm6020.py - DJI Robomaster GM6020 CAN interface library


