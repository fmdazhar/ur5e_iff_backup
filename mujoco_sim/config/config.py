import numpy as np


class PegEnvConfig():
    """Set the configuration for FrankaEnv."""
    # General Environment Configuration
    ENV_CONFIG = {
        "action_scale": np.array([0.005,0.005, 1]),  # Scaling factors for position, orientation, and gripper control
        "control_dt": 0.02,  # Time step for controller updates
        "physics_dt": 0.002,  # Time step for physics simulation
        "time_limit": 10.0,  # Time limit for each episode
        "render_mode": "rgb_array",  # Rendering mode ("human" or "rgb_array")
        "image_obs": False,  # Whether to include image observations
        "seed": 0,  # Random seed
    }

    # UR5e Robot Configuration
    UR5E_CONFIG = {
        "home_position": np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0]),  # Home joint angles
        "reset_position": np.array([-2.3047, -2.0615, 2.6054, -2.1147, -1.5708, 0.8369]),  # Reset joint angles
        "cartesian_bounds": np.array([[0.2, -0.3, 0.0], [0.4, 0.3, 0.5]]),  # Workspace boundaries in Cartesian space
        "sampling_bounds": np.array([[0.25, -0.20], [0.35, 0.20]]),  # Sampling range for port placement
        "xy_randomization": False,  # Randomize port placement
        "randomization_bounds": np.array([[-0.1, -0.1, 0.1], [0.1, 0.1, 0.2]]),  # Randomization bounds for positions
        "reset_tolerance": 0.002,  
    }

    # Controller Configuration
    CONTROLLER_CONFIG = {
        "trans_damping_ratio": 0.996,  # Damping ratio for translational control
        "rot_damping_ratio": 0.286,  # Damping ratio for rotational control
        "error_tolerance_pos": 0.001,  # Position error tolerance
        "error_tolerance_ori": 0.001,  # Orientation error tolerance
        "max_pos_error": 0.01,  # Maximum position error
        "max_ori_error": 0.03,  # Maximum orientation error
        "method": "dynamics",  # Control method ("dynamics", "pinv", "svd", etc.)
        "admittance_control": False,  # Whether to use admittance control
        "inertia_compensation": False,  # Whether to compensate for inertia
        "pos_gains": (100, 100, 100),  # Proportional gains for position control
        # "ori_gains": (12.5, 12.5, 12.5),  # Proportional gains for orientation control
        "max_angvel": 4,  # Maximum angular velocity
        "integration_dt": 0.2,  # Integration time step for controller
        "gravity_compensation": True,  # Whether to compensate for gravity  
    }

    # Rendering Configuration
    RENDERING_CONFIG = {
        "width": 640,  # Rendering width
        "height": 480,  # Rendering height
    }

    # Reward Shaping
    REWARD_CONFIG = {
        "reward_shaping": True,  # Use dense reward shaping
        "dense_reward_weights": {
            "box_target": 8.0,  # Weight for reaching target position
            "gripper_box": 4.0,  # Weight for gripper being close to connector
            "no_floor_collision": 0.25,  # Penalty for floor collisions
            "grasping_reward": 0.25,  # Reward for successful grasp
        },
        "sparse_reward_weights": 12.5,  # Reward for completing the task
        "task_complete_tolerance": 0.002,  # Distance threshold for task completion
    }