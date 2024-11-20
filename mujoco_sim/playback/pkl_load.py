import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import imageio
import pickle as pkl

# Load the data from the existing file
file_path = 'updated_traj.pkl'  # Replace with the actual path to your file
with open(file_path, 'rb') as file:
    data = pkl.load(file)

# Inspect the observation keys
first_observation = data[0]['observations']
print("Keys in observations:", first_observation.keys())

# Verify and visualize the images
front_image = first_observation['front']
front_image = np.squeeze(front_image)  # Shape: (128, 128, 3)
wrist_image = first_observation['wrist']
wrist_image = np.squeeze(wrist_image)  # Shape: (128, 128, 3)

# Display the first front camera image
plt.imshow(front_image)
plt.title("Front Camera Image")
plt.show()

# Display the first wrist camera image
plt.imshow(wrist_image)
plt.title("Wrist Camera Image")
plt.show()

# Segment the data into trajectories using 'dones' flags
trajectories = []
current_trajectory = []
for transition in data:
    current_trajectory.append(transition)
    if transition['dones']:  # End of trajectory
        trajectories.append(current_trajectory)
        current_trajectory = []

# Handle any remaining transitions
if current_trajectory:
    trajectories.append(current_trajectory)

# Create videos for each trajectory from 'front' camera images
for idx, traj in enumerate(trajectories):
    images = []
    for elem in traj:
        img = np.array(elem['observations']['front'])  # Shape: (1, H, W, 3)
        img = np.squeeze(img)  # Shape: (H, W, 3)
        images.append(img)
    video_filename = f'front_camera_trajectory_{idx+1:02d}.mp4'
    with imageio.get_writer(video_filename, fps=30) as writer:
        for img in images:
            img_uint8 = img.astype('uint8')
            writer.append_data(img_uint8)
    print(f"Video saved to {video_filename}")

# Create videos for each trajectory from 'wrist' camera images
for idx, traj in enumerate(trajectories):
    images = []
    for elem in traj:
        img = np.array(elem['observations']['wrist'])  # Shape: (1, H, W, 3)
        img = np.squeeze(img)  # Shape: (H, W, 3)
        images.append(img)
    video_filename = f'wrist_camera_trajectory_{idx+1:02d}.mp4'
    with imageio.get_writer(video_filename, fps=30) as writer:
        for img in images:
            img_uint8 = img.astype('uint8')
            writer.append_data(img_uint8)
    print(f"Video saved to {video_filename}")



# # Assuming 20 demos
# num_demos = 20
# demo_size = len(data) // num_demos

# # Update masks and dones for each trajectory
# for demo_idx in range(num_demos):
#     start_idx = demo_idx * demo_size
#     end_idx = (demo_idx + 1) * demo_size
#     demo_data = data[start_idx:end_idx]

#     for i, transition in enumerate(demo_data):
#         if i == len(demo_data) - 1:  # Last timestep in the demo
#             transition['masks'] = 0.0
#             transition['dones'] = True
#         else:
#             transition['masks'] = 1.0
#             transition['dones'] = False

# # Save the updated data to a new file
# updated_file_path = 'updated_traj.pkl'  # Path for the new file
# with open(updated_file_path, 'wb') as updated_file:
#     pkl.dump(data, updated_file)

# print(f"Updated data saved to {updated_file_path}")



# # Examine the first element
# first_element = data[0]

# if isinstance(first_element, dict):
#     print(f"First element is a dictionary with {len(first_element)} keys:")
#     for key, value in first_element.items():
#         if isinstance(value, np.ndarray):
#             print(f"Key: {key}, Type: ndarray, Shape: {value.shape}")
#         else:
#             print(f"Key: {key}, Type: {type(value)}")
# else:
#     print(f"First element is of type {type(first_element)}")

# # Assuming `data` is already loaded
# first_observation = data[0]['observations']

# # Check the type and shape for each key
# for key in first_observation.keys():
#     value = first_observation[key]
#     value_array = np.array(value)
#     print(f"Key: {key}")
#     print(f"Type: {type(value)}")
#     print(f"Shape: {value_array.shape}")
#     print(f"Data Type: {value_array.dtype}")
#     print("-" * 40)

# # Initialize variables to count trajectories and steps
# num_trajectories = 0
# steps_per_trajectory = []

# current_steps = 0

# # Iterate through the data to count trajectories and steps
# for transition in data:
#     current_steps += 1
#     if transition['dones']:  # End of a trajectory
#         num_trajectories += 1
#         steps_per_trajectory.append(current_steps)
#         current_steps = 0  # Reset step counter for the next trajectory

# # If there are remaining steps after the last trajectory (incomplete data):
# if current_steps > 0:
#     num_trajectories += 1
#     steps_per_trajectory.append(current_steps)

# # Output results
# print(f"Number of trajectories: {num_trajectories}")
# print(f"Number of steps in each trajectory: {steps_per_trajectory}")



# first_demo = data[steps_per_trajectory[0]: 2 * steps_per_trajectory[0]]

# # Extract data for plotting
# rewards = [elem['rewards'] for elem in first_demo]
# actions = [elem['actions'] for elem in first_demo]
# states = [elem['observations']['state'][0] for elem in first_demo]  # Extract 'state' from observations
# masks = [elem['masks'] for elem in first_demo]
# dones = [elem['dones'] for elem in first_demo]

# # Convert to NumPy arrays for easier manipulation
# actions = np.array(actions)
# states = np.array(states)
# rewards = np.array(rewards)
# masks = np.array(masks)
# dones = np.array(dones)

# # Plot rewards over time
# plt.figure(figsize=(10, 6))
# plt.plot(rewards, label="Rewards")
# plt.title("Rewards Over Time (First Demo)")
# plt.xlabel("Timestep")
# plt.ylabel("Reward")
# plt.legend()
# plt.grid()
# plt.show()

# # Plot actions over time
# plt.figure(figsize=(10, 6))
# for i in range(actions.shape[1]):
#     plt.plot(actions[:, i], label=f"Action {i+1}")
# plt.title("Actions Over Time (First Demo)")
# plt.xlabel("Timestep")
# plt.ylabel("Action Value")
# plt.legend()
# plt.grid()
# plt.show()

# # Plot states over time
# plt.figure(figsize=(10, 6))
# for i in range(states.shape[1]):
#     plt.plot(states[:, i], label=f"State {i+1}")
# plt.title("States Over Time (First Demo)")
# plt.xlabel("Timestep")
# plt.ylabel("State Value")
# plt.legend()
# plt.grid()
# plt.show()

# # Plot masks over time as points
# plt.figure(figsize=(10, 6))
# plt.scatter(range(len(masks)), masks, label="Masks", color="blue", alpha=0.7)
# plt.title("Masks Over Time (First Demo)")
# plt.xlabel("Timestep")
# plt.ylabel("Mask Value")
# plt.legend()
# plt.grid()
# plt.show()

# # Plot dones over time as points
# plt.figure(figsize=(10, 6))
# plt.scatter(range(len(dones)), dones, label="Dones", color="red", alpha=0.7)
# plt.title("Dones Over Time (First Demo)")
# plt.xlabel("Timestep")
# plt.ylabel("Done Value")
# plt.legend()
# plt.grid()
# plt.show()