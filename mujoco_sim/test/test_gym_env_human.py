import time
import mujoco
import mujoco.viewer
import numpy as np
import glfw

from mujoco_sim import envs
from mujoco_sim.envs.wrappers import SpacemouseIntervention, ZOnlyWrapper, ObsWrapper, GripperCloseEnv


# glfw init
glfw.init()

# env = envs.ur5ePickCubeGymEnv(action_scale=(0.005,0.005, 1))
env = envs.ur5ePegInHoleGymEnv()
env = GripperCloseEnv(env)
env = SpacemouseIntervention(env)
env = ObsWrapper(env)
env = ZOnlyWrapper(env)

action_spec = env.action_space
controller = env.controller


def sample():
    # a = np.random.uniform(action_spec.low, action_spec.high, action_spec.shape)
    a = np.zeros(action_spec.shape, dtype=action_spec.dtype)
    return a.astype(action_spec.dtype)

m = env.model
d = env.data

reset = False
# KEY_SPACE = 32
KEY_SPACE = 92

action = sample()  # Generate an initial action sample
last_sample_time = time.time()  # Track the last sample time



def key_callback(keycode):
    # print(f"Key pressed: {keycode}")
    if keycode == KEY_SPACE:
        global reset
        reset = True


env.reset()
start_time = time.time()
with mujoco.viewer.launch_passive(m, d, key_callback=key_callback, show_right_ui= False) as viewer:
    start = time.time()
    # env.external_viewer = viewer
    # env.reset()

    while viewer.is_running():
        if reset:
            env.reset()
            action = sample()  # Generate a new action sample
            last_sample_time = time.time()  # Reset the action timer
            reset = False
        else:
            step_start = time.time()

            # Update the action every 3 seconds
            if time.time() - last_sample_time >= 10.0:
                action = sample()  # Generate a new action sample
                last_sample_time = time.time()  # Update the last sample time

            env.step(action)
            viewer.sync()

            time_until_next_step = env.control_dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
