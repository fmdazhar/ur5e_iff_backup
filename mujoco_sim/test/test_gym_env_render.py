import time

import gym
import mujoco
import mujoco.viewer
import numpy as np

import mujoco_sim

env = gym.make("ur5ePickCubeVision-v0", render_mode="human", image_obs=True)
action_spec = env.action_space



def sample():
    # a = np.random.uniform(action_spec.low, action_spec.high, action_spec.shape)
    a = np.zeros(action_spec.shape, dtype=action_spec.dtype)

    return a.astype(action_spec.dtype)


obs, info = env.reset()
frames = []

for i in range(200):
    a = sample()
    obs, rew, done, truncated, info = env.step(a)
    # images = obs["images"]
    # frames.append(np.concatenate((images["front"], images["wrist"]), axis=0))

    if done:
        obs, info = env.reset()

# import imageio

# imageio.mimsave("franka_lift_cube_render_test.mp4", frames, fps=20)
