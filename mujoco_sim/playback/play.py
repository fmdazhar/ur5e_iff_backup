import pickle as pkl
import numpy as np
import time

import mujoco
from mujoco import viewer

# Path to your updated .pkl file
traj_file_path = 'updated_traj.pkl'  # Replace with the actual path
# Path to your MuJoCo XML model file
model_xml_path = 'arena.xml'  # Replace with the actual path

# Load the trajectory data
with open(traj_file_path, 'rb') as file:
    data = pkl.load(file)

# Load the MuJoCo model
model = mujoco.MjModel.from_xml_path(model_xml_path)
data_sim = mujoco.MjData(model)

# Assuming 20 demos
num_demos = 20
demo_size = len(data) // num_demos

def set_simulation_state(data_sim, observation):
    # Extract the state vector
    state_vector = observation['state'][0]  # Adjust if necessary

    # Define the joint names for the robot arm
    joint_names = [f"joint{i}" for i in range(1, 8)]  # Adjust joint names as per your model

    # Get the qpos and qvel addresses for these joints
    joint_qpos_indices = [model.jnt_qposadr[model.joint(joint_name).id] for joint_name in joint_names]
    joint_qvel_indices = [model.jnt_dofadr[model.joint(joint_name).id] for joint_name in joint_names]

    # Extract qpos and qvel from the state vector
    qpos_size = len(joint_qpos_indices)
    qvel_size = len(joint_qvel_indices)
    qpos = state_vector[:qpos_size]
    qvel = state_vector[qpos_size:qpos_size + qvel_size]

    # Initialize the simulation state
    sim_state = data_sim

    # Set qpos and qvel for the specified joints
    sim_state.qpos[joint_qpos_indices] = qpos
    sim_state.qvel[joint_qvel_indices] = qvel

    # If your state_vector includes additional data (e.g., actuator forces), adjust accordingly

    # Forward the simulation to compute derived quantities
    mujoco.mj_forward(model, sim_state)

# Collect frames for each trajectory
trajectories = []

for demo_idx in range(num_demos):
    print(f"Processing trajectory {demo_idx + 1}/{num_demos}")
    start_idx = demo_idx * demo_size
    end_idx = (demo_idx + 1) * demo_size
    demo_data = data[start_idx:end_idx]

    frames = []
    for transition in demo_data:
        observation = transition['observations']
        set_simulation_state(data_sim, observation)

        # Copy the current state to store as a frame
        frame = data_sim.qpos.copy(), data_sim.qvel.copy()
        frames.append(frame)

    trajectories.append(frames)

# Function to play a trajectory using the passive viewer
def play_trajectory(model, trajectory):
    data_sim = mujoco.MjData(model)

    # Create a generator for the trajectory frames
    def trajectory_generator():
        for qpos, qvel in trajectory:
            data_sim.qpos[:] = qpos
            data_sim.qvel[:] = qvel
            mujoco.mj_forward(model, data_sim)
            yield

    # Initialize the viewer with the model and data
    with viewer.launch_passive(model, data_sim) as viewer_instance:
        # Play the trajectory
        for _ in trajectory_generator():
            # The viewer handles rendering; no need to step the simulation
            time.sleep(0.01)  # Control playback speed if necessary

# Play each trajectory one by one
for idx, trajectory in enumerate(trajectories):
    print(f"Playing trajectory {idx + 1}/{num_demos}")
    play_trajectory(model, trajectory)
    input("Press Enter to proceed to the next trajectory...")
