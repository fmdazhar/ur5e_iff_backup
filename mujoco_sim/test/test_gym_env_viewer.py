import time
import mujoco
import mujoco_viewer
import numpy as np

from mujoco_sim import envs
from mujoco_sim.utils.viz import SliderController
from dm_robotics.transformations import transformations as tr


# Initialize the environment and controller
env = envs.ur5ePickCubeGymEnv(action_scale=(1, 1))
action_spec = env.action_space
controller = env.controller
slider_controller = SliderController(controller)

# Set controller parameters dynamically
controller.set_parameters(
    damping_ratio=1,
    error_tolerance_pos=0.01,
    error_tolerance_ori=0.01,
    pos_gains=(1, 1, 1),
    ori_gains=(0.5, 0.5, 0.5),
    method="dls"
)

# Sample a random action within the action space
def sample():
    return np.random.uniform(action_spec.low, action_spec.high, action_spec.shape).astype(action_spec.dtype)

# Environment data and variables
m = env.model
d = env.data
reset = False
KEY_SPACE = 32
action = sample()
last_sample_time = time.time()

# Reset the environment
env.reset()
viewer = mujoco_viewer.MujocoViewer(m, d, hide_menus=True)

# Define indices for UR5e DOF and gripper
ur5e_dof_indices = env._ur5e_dof_ids
gripper_dof_index = env._gripper_ctrl_id

# Set up graph lines for UR5e DOF and gripper
for joint_idx in ur5e_dof_indices:
    viewer.add_line_to_fig(line_name=f"qpos_ur5e_joint_{joint_idx}", fig_idx=0)
viewer.add_line_to_fig(line_name="qpos_gripper", fig_idx=0)

# Set up figure properties for visualization
fig0 = viewer.figs[0]
fig0.title = "UR5e Joint Positions"
fig0.xlabel = "Timesteps"
fig0.flg_legend = True
fig0.figurergba[0] = 0.2
fig0.figurergba[3] = 0.2
fig0.gridsize[0] = 5
fig0.gridsize[1] = 5

# Set up distinct colors for each axis in the target and current position lines
viewer.add_line_to_fig(line_name="target_x", fig_idx=1, color=[0.85, 0.3, 0])  # Dark Orange
viewer.add_line_to_fig(line_name="target_y", fig_idx=1, color=[0.2, 0.6, 0.2])  # Dark Green
viewer.add_line_to_fig(line_name="target_z", fig_idx=1, color=[0.2, 0.4, 0.8])  # Deep Blue

viewer.add_line_to_fig(line_name="current_x", fig_idx=1, color=[1, 0.5, 0.2])  # Light Orange
viewer.add_line_to_fig(line_name="current_y", fig_idx=1, color=[0.4, 0.9, 0.4])  # Light Green
viewer.add_line_to_fig(line_name="current_z", fig_idx=1, color=[0.3, 0.6, 1])  # Lighter Blue

fig1 = viewer.figs[1]
fig1.title = "End-Effector Position Tracking"
fig1.xlabel = "Timesteps"
fig1.flg_legend = True
fig1.figurergba[0] = 0.2
fig1.figurergba[3] = 0.2
fig1.gridsize[0] = 5
fig1.gridsize[1] = 5

# Figure 2: TCP Velocity
viewer.add_line_to_fig(line_name="tcp_vel_x", fig_idx=2, color=[0.8, 0.1, 0.1])  # Red
viewer.add_line_to_fig(line_name="tcp_vel_y", fig_idx=2, color=[0.1, 0.8, 0.1])  # Green
viewer.add_line_to_fig(line_name="tcp_vel_z", fig_idx=2, color=[0.1, 0.1, 0.8])  # Blue

fig2 = viewer.figs[2]
fig2.title = "TCP Velocity"
fig2.xlabel = "Timesteps"
fig2.flg_legend = True
fig2.figurergba[0] = 0.2
fig2.figurergba[3] = 0.2
fig2.gridsize[0] = 5
fig2.gridsize[1] = 5

# Figure 3: Joint Velocity
for joint_idx in ur5e_dof_indices:
    viewer.add_line_to_fig(line_name=f"joint_vel_{joint_idx}", fig_idx=3)

fig3 = viewer.figs[3]
fig3.title = "Joint Velocities"
fig3.xlabel = "Timesteps"
fig3.flg_legend = True
fig3.figurergba[0] = 0.2
fig3.figurergba[3] = 0.2
fig3.gridsize[0] = 5
fig3.gridsize[1] = 5

# Figure 4: Joint Torques
for joint_idx in ur5e_dof_indices:
    viewer.add_line_to_fig(line_name=f"joint_torque_{joint_idx}", fig_idx=4)

fig4 = viewer.figs[4]
fig4.title = "Joint Torques"
fig4.xlabel = "Timesteps"
fig4.flg_legend = True
fig4.figurergba[0] = 0.2
fig4.figurergba[3] = 0.2
fig4.gridsize[0] = 5
fig4.gridsize[1] = 5

# Figure 5: Wrist Force
viewer.add_line_to_fig(line_name="wrist_force_x", fig_idx=5, color=[0.6, 0.1, 0.1])  # Dark Red
viewer.add_line_to_fig(line_name="wrist_force_y", fig_idx=5, color=[0.1, 0.6, 0.1])  # Dark Green
viewer.add_line_to_fig(line_name="wrist_force_z", fig_idx=5, color=[0.1, 0.1, 0.6])  # Dark Blue

fig5 = viewer.figs[5]
fig5.title = "Wrist Force"
fig5.xlabel = "Timesteps"
fig5.flg_legend = True
fig5.figurergba[0] = 0.2
fig5.figurergba[3] = 0.2
fig5.gridsize[0] = 5
fig5.gridsize[1] = 5

# Main simulation loop
while viewer.is_alive:
    if viewer.reset_requested:
        env.reset()
        action = sample()
        last_sample_time = time.time()
        viewer.reset_requested = False  # Reset the flag

    else:
        step_start = time.time()

        # Update action every 3 seconds
        if time.time() - last_sample_time >= 3.0:
            action = sample()
            last_sample_time = time.time()

        env.step(action)

        # Add marker at mocap position
        mocap_pos = d.mocap_pos[0]
        tcp_pos = d.site_xpos[controller.site_id]
        rotation_matrix = tr.quat_to_mat(d.mocap_quat[0])[:3, :3]

        viewer.add_marker(
            pos=mocap_pos,
            mat=rotation_matrix,
            size=[0.01, 0.01, 0.2],
            rgba=[0, 1, 1, 0.6],
            type=mujoco.mjtGeom.mjGEOM_ARROW,
        )

        # Update graph lines for UR5e DOF
        for joint_idx in ur5e_dof_indices:
            viewer.add_data_to_line(line_name=f"qpos_ur5e_joint_{joint_idx}", line_data=d.qpos[joint_idx], fig_idx=0)

        # Update gripper DOF line
        viewer.add_data_to_line(line_name="qpos_gripper", line_data=d.ctrl[gripper_dof_index] / 255, fig_idx=0)

        # Update target and current position lines
        viewer.add_data_to_line(line_name="target_x", line_data=mocap_pos[0], fig_idx=1)
        viewer.add_data_to_line(line_name="target_y", line_data=mocap_pos[1], fig_idx=1)
        viewer.add_data_to_line(line_name="target_z", line_data=mocap_pos[2], fig_idx=1)
        
        viewer.add_data_to_line(line_name="current_x", line_data=tcp_pos[0], fig_idx=1)
        viewer.add_data_to_line(line_name="current_y", line_data=tcp_pos[1], fig_idx=1)
        viewer.add_data_to_line(line_name="current_z", line_data=tcp_pos[2], fig_idx=1)

        # Update TCP velocity lines
        tcp_vel = d.sensor("hande/pinch_vel").data
        viewer.add_data_to_line(line_name="tcp_vel_x", line_data=tcp_vel[0], fig_idx=2)
        viewer.add_data_to_line(line_name="tcp_vel_y", line_data=tcp_vel[1], fig_idx=2)
        viewer.add_data_to_line(line_name="tcp_vel_z", line_data=tcp_vel[2], fig_idx=2)

        # Update Joint Velocity lines
        for joint_idx in ur5e_dof_indices:
            viewer.add_data_to_line(line_name=f"joint_vel_{joint_idx}", line_data=d.qvel[joint_idx], fig_idx=3)

        # Update Joint Torque lines
        for joint_idx in ur5e_dof_indices:
            viewer.add_data_to_line(line_name=f"joint_torque_{joint_idx}", line_data=d.qfrc_actuator[joint_idx], fig_idx=4)

        # Update Wrist Force lines
        wrist_force = d.sensor("ur5e/wrist_force").data
        viewer.add_data_to_line(line_name="wrist_force_x", line_data=wrist_force[0], fig_idx=5)
        viewer.add_data_to_line(line_name="wrist_force_y", line_data=wrist_force[1], fig_idx=5)
        viewer.add_data_to_line(line_name="wrist_force_z", line_data=wrist_force[2], fig_idx=5)

        viewer.render()

        # Update Tkinter sliders
        slider_controller.root.update_idletasks()
        slider_controller.root.update()

        # Control timestep synchronization
        time_until_next_step = env.control_dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

# Close viewer after simulation ends
viewer.close()