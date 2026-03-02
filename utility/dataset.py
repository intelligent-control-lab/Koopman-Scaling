import torch
import numpy as np
import os
from torch.utils.data import Dataset
import pandas as pd
import pybullet as pb
import pybullet_data
from tqdm import tqdm
from scipy.integrate import odeint
import random


def _get_project_root():
    """Get absolute path to Koopman-Scaling project root."""
    # This file is in utility/, so project root is one level up
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)

def _get_franka_urdf_path():
    """Get absolute path to Franka URDF file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(script_dir, "franka_description", "robots", "franka_panda.urdf")
    return urdf_path

class PolynomialDataCollector:
    def __init__(self, state_dim=3, m=100, a1=0.85, a2=0.9, a3=0.90, b=None):
        self.state_dim = state_dim
        self.m = m
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        if b is None:
            self.b = np.linspace(0.9, 0.1, m-2)
        else:
            self.b = b

    def random_state(self):
        return np.random.uniform(-1, 1, size=(self.state_dim,)).astype(np.float64)
    
    def simulate_dynamics(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        poly_sum = np.zeros_like(x1)
        for p in range(1, self.m-1):
            poly_sum += self.b[p-1] * (x1 ** p)
        x1_next = self.a1 * x1
        x2_next = self.a2 * x2
        x3_next = self.a3 * x3 + poly_sum
        x_next = np.stack([x1_next, x2_next, x3_next], axis=1)
        return x_next

    def collect_koopman_data(self, traj_num, steps):
        data = np.empty((steps + 1, traj_num, self.state_dim))
        for traj_i in range(traj_num):
            x0 = self.random_state()
            data[0, traj_i, :] = x0
            current_state = x0
            for i in range(1, steps + 1):
                next_state = self.simulate_dynamics(current_state.reshape(1, -1))[0]
                data[i, traj_i, :] = next_state
                current_state = next_state
        return data

class LogisticMapDataCollector:
    def __init__(self, state_dim=1, lambda_param=3.8):
        self.state_dim = state_dim
        if lambda_param is not None:
            self.lambda_param = lambda_param
        else:
            self.lambda_param = np.random.uniform(3.5, 4.0)
        
    def random_state(self):
        return np.random.uniform(0, 1, size=(self.state_dim,)).astype(np.float64)
    
    def simulate_dynamics(self, x):
        return self.lambda_param * x * (1 - x)
    
    def collect_koopman_data(self, traj_num, steps):
        data = np.empty((steps + 1, traj_num, self.state_dim))
        for traj_i in range(traj_num):
            x0 = self.random_state()
            data[0, traj_i, :] = x0
            current_state = x0
            for i in range(1, steps + 1):
                next_state = self.simulate_dynamics(current_state.reshape(1, -1))[0]
                data[i, traj_i, :] = next_state
                current_state = next_state
        return data

class DampingPendulumDataCollector:
    def __init__(self):
        self.g = 9.8
        self.l = 1.0
        self.m = 1.0
        self.b = 1.0
        self.dt = 0.02
        self.state_dim = 2
        self.u_dim = 1
        self.umin = -8.0
        self.umax =  8.0

    def _dynamics(self, y, t, u):
        theta, dtheta = y
        ddtheta = (
            - self.g/self.l * np.sin(theta)
            - self.b*self.l*dtheta/self.m
            + (np.cos(theta)*u)/(self.m*self.l)
        )
        return [dtheta, ddtheta]

    def random_state(self):
        theta  = random.uniform(-2*np.pi, 2*np.pi)
        dtheta = random.uniform(-8.0, 8.0)
        return np.array([theta, dtheta], dtype=np.float64)

    def random_control(self):
        u = random.uniform(self.umin, self.umax)
        return np.array([u], dtype=np.float64)

    def simulate_dynamics(self, state, control):
        sol = odeint(self._dynamics, state, [0.0, self.dt], args=(control[0],))
        return sol[-1].astype(np.float64)

    def collect_koopman_data(self, traj_num, steps):
        data = np.empty((steps + 1, traj_num, self.state_dim + self.u_dim),
                        dtype=np.float64)

        for traj in tqdm(range(traj_num)):
            s = self.random_state()
            u = self.random_control()
            data[0, traj, :] = np.concatenate([u, s])

            for t in range(1, steps + 1):
                u = self.random_control()
                s = self.simulate_dynamics(s, u)
                data[t, traj, :] = np.concatenate([u, s])

        return data

class DoublePendulumDataCollector:
    def __init__(self):
        self.g  = 9.8
        self.l1 = 1.0
        self.l2 = 1.0
        self.m1 = 1.0
        self.m2 = 1.0
        self.dt = 0.01
        self.state_dim = 4
        self.u_dim = 2
        self.umin = np.array([-6.0, -6.0], dtype=np.float64)
        self.umax = np.array([ 6.0,  6.0], dtype=np.float64)

    def _dynamics(self, y, t, u1, u2):
        th1, th2, dth1, dth2 = y
        g, l1, l2, m1, m2 = self.g, self.l1, self.l2, self.m1, self.m2
        c2 = np.cos(th2)
        s2 = np.sin(th2)

        M11 = m1*l1**2 + m2*(l1**2 + 2*l1*l2*c2 + l2**2)
        M12 = m2*(l1*l2*c2 + l2**2)
        M21 = M12
        M22 = m2*l2**2
        M = np.array([[M11, M12], [M21, M22]], dtype=np.float64)

        C1 = -m2*l1*l2*s2*(2*dth1*dth2 + dth2**2)
        C2 =  m2*l1*l2*dth1**2 * s2
        C = np.array([C1, C2], dtype=np.float64)

        G1 = (m1+m2)*l1*g*np.cos(th1) + m2*l2*g*np.cos(th1 + th2)
        G2 = m2*l2*g*np.cos(th1 + th2)
        G = np.array([G1, G2], dtype=np.float64)

        tau = np.array([u1, u2], dtype=np.float64)
        dd = np.linalg.pinv(M).dot(tau - C - G)

        return [dth1, dth2, dd[0], dd[1]]

    def random_state(self):
        th1  = random.uniform(-0.1*np.pi, 0.1*np.pi)
        dth1 = random.uniform(-1.0, 1.0)
        th2  = random.uniform(-0.1*np.pi, 0.1*np.pi)
        dth2 = random.uniform(-1.0, 1.0)
        return np.array([th1, th2, dth1, dth2], dtype=np.float64)

    def random_control(self):
        u1 = random.uniform(self.umin[0], self.umax[0])
        u2 = random.uniform(self.umin[1], self.umax[1])
        return np.array([u1, u2], dtype=np.float64)

    def simulate_dynamics(self, state, control):
        sol = odeint(self._dynamics, state, [0.0, self.dt],
                     args=(control[0], control[1]))
        return sol[-1].astype(np.float64)

    def collect_koopman_data(self, traj_num, steps):
        data = np.empty((steps + 1, traj_num, self.state_dim + self.u_dim),
                        dtype=np.float64)

        for traj in tqdm(range(traj_num)):
            s = self.random_state()
            u = self.random_control()
            data[0, traj, :] = np.concatenate([u, s])

            for t in range(1, steps + 1):
                u = self.random_control()
                s = self.simulate_dynamics(s, u)
                data[t, traj, :] = np.concatenate([u, s])

        return data

class FrankaDataCollector:
    """
    Improved data collector for Franka robot dynamics learning.

    Key improvements over the original FrankaDataCollector:
    - ~2x larger velocity range (0.04-0.30 vs 0.12 rad/s)
    - Full joint space initialization (vs +/-0.2 rad from home)
    - 3 trajectory types: goal-reaching, smooth random walk, sinusoidal
    - Temporal correlation in actions (vs iid noise)

    Trajectory types:
    - goal_reaching (40%): Minimum-jerk point-to-point motion
    - smooth_random (40%): Ornstein-Uhlenbeck correlated random walk
    - sinusoidal (20%): Periodic oscillations for frequency coverage
    """

    # Joint limits from Franka URDF
    JOINT_LOW = np.array([-2.9, -1.8, -2.9, -3.0, -2.9, -0.08, -2.9], dtype=np.float32)
    JOINT_HIGH = np.array([2.9, 1.8, 2.9, 0.08, 2.9, 3.0, 2.9], dtype=np.float32)

    # Per-joint velocity limits (rad/s)
    VELOCITY_LIMITS = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    # Velocity regimes
    VELOCITY_REGIMES = {
        'slow': (0.04, 0.12),
        'medium': (0.12, 0.20),
        'fast': (0.20, 0.30),
    }
    REGIME_WEIGHTS = np.array([0.30, 0.40, 0.30])

    def __init__(self, render=False, ts=0.002, seed=42):
        """Initialize Franka data collector with PyBullet simulation."""
        self.frame_skip = 10
        self.dt = self.frame_skip * ts
        self.ts = ts

        # PyBullet setup
        self.client = pb.connect(pb.GUI if render else pb.DIRECT)
        pb.setTimeStep(ts)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.loadURDF("plane.urdf")
        self.robot = pb.loadURDF(
            _get_franka_urdf_path(),
            [0.0, 0.0, 0.0], useFixedBase=1,
        )
        pb.setGravity(0, 0, -9.81)

        # Dimensions
        self.Nstates = 14  # 7 positions + 7 velocities
        self.udim = 7
        self.n_joints = 7
        self.sat_val = 1.0

        # Random state
        self.rng = np.random.default_rng(seed)
        self.home_position = np.array([0.0, -0.78, 0.0, -2.35, 0.0, 1.57, 0.78], dtype=np.float32)

        self.reset()

    def close(self):
        try:
            pb.disconnect(self.client)
        except Exception:
            pass

    def reset(self):
        for i, jnt in enumerate(self.home_position):
            pb.resetJointState(self.robot, i, float(jnt))
        return self.get_state()

    def reset_state(self, joint_positions):
        joint_positions = np.clip(joint_positions, self.JOINT_LOW, self.JOINT_HIGH)
        for i in range(self.n_joints):
            pb.resetJointState(self.robot, i, float(joint_positions[i]))
        return self.get_state()

    def get_state(self):
        """Get current state: [q(7), dq(7)] -> (14,)."""
        jnt_st = pb.getJointStates(self.robot, range(self.n_joints))
        q = np.array([s[0] for s in jnt_st], dtype=np.float32)
        dq = np.array([s[1] for s in jnt_st], dtype=np.float32)
        return np.concatenate([q, dq], axis=0)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -self.sat_val, self.sat_val)
        pb.setJointMotorControlArray(
            self.robot, range(self.n_joints), pb.VELOCITY_CONTROL,
            targetVelocities=a.tolist(),
        )
        for _ in range(self.frame_skip):
            pb.stepSimulation()
        return self.get_state()

    def _sample_initial_state(self):
        """Sample random joint configuration within limits."""
        margin = 0.3
        return self.rng.uniform(
            self.JOINT_LOW + margin, self.JOINT_HIGH - margin
        ).astype(np.float32)

    def _generate_goal_reaching(self, steps, current_pos):
        """Minimum-jerk trajectory to random goal."""
        margin = 0.3
        goal = self.rng.uniform(
            self.JOINT_LOW + margin, self.JOINT_HIGH - margin,
            size=self.n_joints
        )
        displacement = goal - current_pos
        T = steps * self.dt
        velocities = np.zeros((steps, self.n_joints), dtype=np.float32)

        for i in range(steps):
            tau = (i + 0.5) * self.dt / T
            s_dot = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
            velocities[i, :] = displacement * s_dot / T

        return velocities

    def _generate_smooth_random(self, steps):
        """Ornstein-Uhlenbeck process for smooth random walk."""
        theta = self.rng.uniform(0.2, 0.5)
        sigma = self.rng.uniform(0.8, 1.5)
        velocities = np.zeros((steps, self.n_joints), dtype=np.float32)
        v = self.rng.uniform(-0.3, 0.3, size=self.n_joints).astype(np.float32)
        sqrt_dt = np.sqrt(self.dt)

        for i in range(steps):
            dW = self.rng.standard_normal(self.n_joints).astype(np.float32) * sqrt_dt
            v = v + theta * (0 - v) * self.dt + sigma * dW
            velocities[i, :] = v

        return velocities

    def _generate_sinusoidal(self, steps):
        """Multi-frequency sinusoidal oscillations."""
        frequencies = self.rng.uniform(0.2, 1.5, size=self.n_joints)
        amplitudes = self.rng.uniform(0.5, 1.5, size=self.n_joints)
        phases = self.rng.uniform(0, 2 * np.pi, size=self.n_joints)
        t = np.arange(steps) * self.dt
        velocities = np.zeros((steps, self.n_joints), dtype=np.float32)

        for j in range(self.n_joints):
            velocities[:, j] = amplitudes[j] * np.sin(2 * np.pi * frequencies[j] * t + phases[j])

        return velocities

    def _scale_velocities(self, velocities):
        """Scale velocities to target regime and clip to limits."""
        regimes = list(self.VELOCITY_REGIMES.keys())
        regime = self.rng.choice(regimes, p=self.REGIME_WEIGHTS)
        v_min, v_max = self.VELOCITY_REGIMES[regime]
        target_scale = self.rng.uniform(v_min, v_max)

        current_max = np.max(np.abs(velocities))
        if current_max > 1e-6:
            velocities = velocities * (target_scale / current_max)

        for j in range(self.n_joints):
            velocities[:, j] = np.clip(
                velocities[:, j], -self.VELOCITY_LIMITS[j], self.VELOCITY_LIMITS[j]
            )
        return velocities.astype(np.float32)

    def _generate_trajectory(self, steps, current_pos):
        """Generate velocity trajectory using one of 3 strategies."""
        traj_type = self.rng.choice(
            ['goal_reaching', 'smooth_random', 'sinusoidal'],
            p=[0.40, 0.40, 0.20]
        )

        if traj_type == 'goal_reaching':
            velocities = self._generate_goal_reaching(steps, current_pos)
        elif traj_type == 'smooth_random':
            velocities = self._generate_smooth_random(steps)
        else:
            velocities = self._generate_sinusoidal(steps)

        return self._scale_velocities(velocities)

    def collect_koopman_data(self, traj_num, steps):
        """
        Collect trajectory data for Koopman learning.

        Returns:
            data: np.ndarray of shape (steps + 1, traj_num, 21)
                  Format: [action(7), state(14)] at each timestep
        """
        data = np.empty((steps + 1, traj_num, self.Nstates + self.udim), dtype=np.float32)

        for traj_i in tqdm(range(traj_num), desc="Collecting Franka trajectories"):
            init_joints = self._sample_initial_state()
            state = self.reset_state(init_joints)
            velocities = self._generate_trajectory(steps, state[:self.n_joints])

            action = velocities[0] if len(velocities) > 0 else np.zeros(self.udim, dtype=np.float32)
            data[0, traj_i, :] = np.concatenate([action, state])

            for t in range(1, steps + 1):
                state = self.step(action)
                action = velocities[t] if t < steps else np.zeros(self.udim, dtype=np.float32)
                data[t, traj_i, :] = np.concatenate([action, state])

        return data

class G1Go2DataCollector():
    def __init__(self, env_name, use_initial_data=False):
        self.use_initial_data = use_initial_data
        project_root = _get_project_root()

        if use_initial_data:
            g1_initial_path = 'None_trajnum90000_trajlen100'
            go2_initial_path = 'None_trajnum89947_trajlen100'
            if env_name == 'Go2':
                initial_dataset_path = os.path.join(project_root, "data", "unitree_go2_flat", "initial_dataset", f"{go2_initial_path}.npz")
            elif env_name == 'G1':
                initial_dataset_path = os.path.join(project_root, "data", "g1_flat", "initial_dataset", f"{g1_initial_path}.npz")
            else:
                raise ValueError("Dataset not found for the given environment.")
            self.data_paths = [initial_dataset_path]
        else:
            self.data_paths = []

        go2_tracking_path_0 = '2025-03-24-20-45-16_trajnum30000_trajlen15'
        go2_tracking_path_1 = '2025-03-24-21-14-03_trajnum30000_trajlen15'
        go2_tracking_path_2 = '2025-03-24-21-57-32_trajnum30000_trajlen15'
        go2_tracking_path_3 = '2025-03-24-22-46-11_trajnum30000_trajlen15'
        g1_tracking_path_0 = '2025-03-23-23-31-06_trajnum30000_trajlen15'
        g1_tracking_path_1 = '2025-03-23-23-59-32_trajnum30000_trajlen15'
        g1_tracking_path_2 = '2025-03-24-00-43-16_trajnum30000_trajlen15'
        g1_tracking_path_3 = '2025-03-24-01-32-42_trajnum30000_trajlen15'
        g1_tracking_path_4 = '2025-03-24-02-38-25_trajnum30000_trajlen15'
        g1_tracking_path_5 = '2025-03-24-04-01-44_trajnum30000_trajlen15'
        if env_name == 'Go2':
            go2_data_dir = os.path.join(project_root, "data", "unitree_go2_flat", "tracking_dataset")
            tracking_dataset_path_0 = os.path.join(go2_data_dir, f"{go2_tracking_path_0}.npz")
            tracking_dataset_path_1 = os.path.join(go2_data_dir, f"{go2_tracking_path_1}.npz")
            tracking_dataset_path_2 = os.path.join(go2_data_dir, f"{go2_tracking_path_2}.npz")
            tracking_dataset_path_3 = os.path.join(go2_data_dir, f"{go2_tracking_path_3}.npz")
            self.data_paths = self.data_paths + [tracking_dataset_path_0, tracking_dataset_path_1, tracking_dataset_path_2, tracking_dataset_path_3]
        elif env_name == 'G1':
            g1_data_dir = os.path.join(project_root, "data", "g1_flat", "tracking_dataset")
            tracking_dataset_path_0 = os.path.join(g1_data_dir, f"{g1_tracking_path_0}.npz")
            tracking_dataset_path_1 = os.path.join(g1_data_dir, f"{g1_tracking_path_1}.npz")
            tracking_dataset_path_2 = os.path.join(g1_data_dir, f"{g1_tracking_path_2}.npz")
            tracking_dataset_path_3 = os.path.join(g1_data_dir, f"{g1_tracking_path_3}.npz")
            tracking_dataset_path_4 = os.path.join(g1_data_dir, f"{g1_tracking_path_4}.npz")
            tracking_dataset_path_5 = os.path.join(g1_data_dir, f"{g1_tracking_path_5}.npz")
            self.data_paths = self.data_paths + [tracking_dataset_path_0, tracking_dataset_path_1, tracking_dataset_path_2, tracking_dataset_path_3, tracking_dataset_path_4, tracking_dataset_path_5]
        else:
            raise ValueError("Dataset not found for the given environment.")
    
    def get_data(self, data_paths, steps=15):
        state_data = []
        action_data = []
        n_initial = 60000
        for i, path in enumerate(data_paths):
            if self.use_initial_data and i == 0:
                state_data.append(np.load(path)['state_data'][:, :n_initial, :])
                action_data.append(np.load(path)['action_data'][:, :n_initial, :])
            else:
                # use all samples
                state_data.append(np.load(path)['state_data'])
                action_data.append(np.load(path)['action_data'])

            if state_data[-1].shape[0] != steps+1:
                state_data[-1] = state_data[-1][:steps+1, :, :]
                action_data[-1] = action_data[-1][:steps, :, :]
        state_data = np.concatenate(state_data, axis=1)
        action_data = np.concatenate(action_data, axis=1)
        num_traj = state_data.shape[1]
        T = state_data.shape[0]
        state_dim = state_data.shape[2]
        action_dim = action_data.shape[2]
        combined_data = np.empty((T, num_traj, state_dim+action_dim), dtype=state_data.dtype)
        for t in range(T-1):
            combined_data[t, :, :] = np.concatenate([action_data[t], state_data[t]], axis=-1)
        combined_data[T-1, :, :] = np.concatenate([np.zeros((num_traj, action_dim), dtype=state_data.dtype), state_data[T-1]], axis=-1)
        return combined_data
    
    def collect_koopman_data(self, traj_num, steps):
        return self.get_data(self.data_paths, steps)[:, :traj_num, :]

class KinovaDataCollector():
    def __init__(self):
        self.state_dim = 14
        self.u_dim = 7
        self.data_paths = ['output_20250402_172619.txt',
                            'output_20250402_182836.txt',
                            'output_20250402_195709.txt',
                            'output_20250402_205831.txt',
                            'output_20250403_104412.txt']

    def get_data(self, data_paths, steps=10):
        kinova_data_dir = os.path.join(_get_project_root(), "data", "kinova_data")
        def process_data(file_path):
            df = pd.read_csv(os.path.join(kinova_data_dir, file_path),
                        delimiter=' ',
                        header=None,
                        on_bad_lines='skip',
                        engine='python')
            arr = df.to_numpy()
            total_data = arr.shape[0]
            trimmed_len = (total_data // steps) * steps
            trimmed = arr[:trimmed_len]
            return trimmed.reshape(steps, -1, arr.shape[1])
        lst = []
        for path in data_paths:
            lst.append(process_data(path))
        return np.concatenate(lst, axis=1)

    def collect_koopman_data(self, traj_num, steps):
        return self.get_data(self.data_paths, steps)[:, :traj_num, :]

def trim_robot_states(env_name, state_data, action_data):
    """
    Trim robot states to exclude hands and extract relevant components.
    Based on the original mpc_tracking.py logic.
    """
    if env_name == "G1":
        # G1: Extract [joint_pos[:23], joint_vel[37:60], height[76:77], root_state[81:]]
        # joint_pos: first 23 DOFs (exclude hands)
        # joint_vel: positions 37-59 (23 velocities corresponding to first 23 positions)
        # height: position 76 (z-coordinate)  
        # root_state: positions 81-86 (6D root state: lin_vel[3] + ang_vel[3])
        trimmed_states = np.concatenate([
            state_data[..., :23],        # joint positions (23)
            state_data[..., 37:60],      # joint velocities (23)  
            state_data[..., 76:77],      # height (1)
            state_data[..., 81:]         # root state (6)
        ], axis=-1)
        # G1: Use first 23 actions (exclude hand actions)
        trimmed_actions = action_data[..., :23]
        
    elif env_name == "Go2":
        # Go2: Extract [joint_states[:24], root_states[26:]]
        # joint_states: positions 0-23 (12 pos + 12 vel)
        # root_states: positions 26+ (remaining root state components)
        trimmed_states = np.concatenate([
            state_data[..., :24],        # joint pos + vel (24)
            state_data[..., 26:]         # root state (remaining)
        ], axis=-1)
        # Go2: Use all actions
        trimmed_actions = action_data
        
    else:
        # Other robots: no trimming
        trimmed_states = state_data
        trimmed_actions = action_data
        
    return trimmed_states, trimmed_actions

class KoopmanDatasetCollector():
    def __init__(self, env_name, train_samples=60000, val_samples=20000, test_samples=20000, Ksteps=15, normalize=True, m=100, seed=42):
        np.random.seed(seed)
        random.seed(seed)

        self.normalize = normalize
        self.norm_stats = None

        datasets_dir = os.path.join(_get_project_root(), "data", "datasets")
        os.makedirs(datasets_dir, exist_ok=True)
        norm_str = "norm" if self.normalize else "nonorm"
        if env_name == "Polynomial":
            data_path = os.path.join(datasets_dir, f"dataset_{env_name}_{norm_str}_m_{m}_Ktrain_{train_samples}_Kval_{val_samples}_Ktest_{test_samples}_Ksteps_{Ksteps}.pt")
        else:
            data_path = os.path.join(datasets_dir, f"dataset_{env_name}_{norm_str}_Ktrain_{train_samples}_Kval_{val_samples}_Ktest_{test_samples}_Ksteps_{Ksteps}.pt")

        self.u_dim = None
        self.state_dim = None

        if env_name == "Polynomial":
            collector = PolynomialDataCollector(m=m)
            self.state_dim = collector.state_dim
        elif env_name == "LogisticMap":
            collector = LogisticMapDataCollector()
            self.state_dim = collector.state_dim
        elif env_name == "Franka":
            collector = FrankaDataCollector(seed=seed)
            self.state_dim = collector.Nstates
            self.u_dim = collector.udim
        elif env_name == "DoublePendulum":
            collector = DoublePendulumDataCollector()
            self.state_dim = collector.state_dim
            self.u_dim = collector.u_dim
        elif env_name == "DampingPendulum":
            collector = DampingPendulumDataCollector()
            self.state_dim = collector.state_dim
            self.u_dim = collector.u_dim
        elif env_name == "G1":
            collector = G1Go2DataCollector(env_name, use_initial_data=True)
            self.full_state_dim = 87  # Original state dimension
            self.full_u_dim = 37      # Original action dimension
            # G1 trimmed: 23 joint_pos + 23 joint_vel + 1 height + 6 root_state = 53
            self.state_dim = 53
            self.u_dim = 23           # Trimmed action dimension
        elif env_name == "Go2":
            collector = G1Go2DataCollector(env_name, use_initial_data=True)
            self.full_state_dim = 37  # Original state dimension  
            self.full_u_dim = 12      # Original action dimension
            # Go2 trimmed: 24 joint_states + remaining_root = 24 + (37-26) = 24 + 11 = 35
            self.state_dim = 35
            self.u_dim = 12           # All actions for Go2
        elif env_name == "Kinova":
            collector = KinovaDataCollector()
            self.state_dim = collector.state_dim
            self.u_dim = collector.u_dim
        else:
            raise ValueError("Unknown environment name.")
        
        if not os.path.exists(data_path):
            data = collector.collect_koopman_data(train_samples+val_samples+test_samples, Ksteps)
            if hasattr(collector, 'close'):
                collector.close()
            
            # Apply state trimming for G1 and Go2
            if env_name in ["G1", "Go2"]:
                print(f"[INFO] Original data shape: {data.shape}")
                # Extract states and actions from combined data
                if self.u_dim is not None:
                    original_states = data[:-1, :, self.full_u_dim:]  # Skip last timestep for states
                    original_actions = data[:-1, :, :self.full_u_dim]  # Actions for all timesteps except last
                    last_states = data[-1:, :, self.full_u_dim:]       # Last timestep state
                    
                    # Apply trimming
                    trimmed_states, trimmed_actions = trim_robot_states(env_name, original_states, original_actions)
                    trimmed_last_states, _ = trim_robot_states(env_name, last_states, np.zeros_like(last_states[..., :self.u_dim]))
                    
                    # Reconstruct data with trimmed dimensions
                    T, N, _ = data.shape
                    new_data = np.empty((T, N, self.u_dim + self.state_dim), dtype=data.dtype)
                    
                    # Fill in the trimmed data
                    for t in range(T-1):
                        new_data[t, :, :] = np.concatenate([trimmed_actions[t], trimmed_states[t]], axis=-1)
                    # Last timestep: zero actions + last state
                    new_data[T-1, :, :] = np.concatenate([np.zeros((N, self.u_dim)), trimmed_last_states[0]], axis=-1)
                    
                    data = new_data
                    print(f"[INFO] Trimmed data shape: {data.shape}")
                    print(f"[INFO] New state_dim: {self.state_dim}, new u_dim: {self.u_dim}")

            permutation = np.random.permutation(data.shape[1])
            shuffled = data[:, permutation, :]

            train_data = shuffled[:, :train_samples, :]
            val_data = shuffled[:, train_samples:train_samples+val_samples, :]
            test_data = shuffled[:, train_samples+val_samples:train_samples+val_samples+test_samples, :]
            
            if self.normalize:
                if self.u_dim is None:
                    train_mean = np.mean(train_data, axis=(0,1))
                    train_std = np.std(train_data, axis=(0,1))
                    train_data = (train_data - train_mean) / train_std
                    val_data = (val_data - train_mean) / train_std
                    test_data = (test_data - train_mean) / train_std
                else:
                    action_train_mean = np.mean(train_data[..., :self.u_dim], axis=(0,1))
                    action_train_std = np.std(train_data[..., :self.u_dim], axis=(0,1))
                    state_train_mean = np.mean(train_data[..., self.u_dim:], axis=(0,1))
                    state_train_std = np.std(train_data[..., self.u_dim:], axis=(0,1))

                    action_train_std = np.maximum(action_train_std, 1e-8)
                    state_train_std = np.maximum(state_train_std, 1e-8)

                    train_data[..., :self.u_dim] = (train_data[..., :self.u_dim] - action_train_mean) / (action_train_std)
                    train_data[..., self.u_dim:] = (train_data[..., self.u_dim:] - state_train_mean) / (state_train_std)
                    val_data[..., :self.u_dim] = (val_data[..., :self.u_dim] - action_train_mean) / (action_train_std)
                    val_data[..., self.u_dim:] = (val_data[..., self.u_dim:] - state_train_mean) / (state_train_std)
                    test_data[..., :self.u_dim] = (test_data[..., :self.u_dim] - action_train_mean) / (action_train_std)
                    test_data[..., self.u_dim:] = (test_data[..., self.u_dim:] - state_train_mean) / (state_train_std)

                    self.norm_stats = {
                        'action_mean': action_train_mean,   # shape (u_dim_trim,)
                        'action_std':  action_train_std,
                        'state_mean':  state_train_mean,    # shape (state_dim_trim,)
                        'state_std':   state_train_std,
                    }
            
            # torch.save({"Ktrain_data": train_data, "Kval_data": val_data, "Ktest_data": test_data}, data_path)
            torch.save({
                "Ktrain_data": train_data,
                "Kval_data":   val_data,
                "Ktest_data":  test_data,
                "norm_stats":  self.norm_stats if self.normalize and self.u_dim is not None else None,
            }, data_path)

        # self.train_data = torch.load(data_path, weights_only=False)["Ktrain_data"]
        # self.val_data = torch.load(data_path, weights_only=False)["Kval_data"]
        # self.test_data = torch.load(data_path, weights_only=False)["Ktest_data"]
        loaded = torch.load(data_path, weights_only=False)
        self.train_data = loaded["Ktrain_data"]
        self.val_data   = loaded["Kval_data"]
        self.test_data  = loaded["Ktest_data"]
        self.norm_stats = loaded.get("norm_stats", None)

    
    def get_data(self):
        return self.train_data, self.val_data, self.test_data

class KoopmanDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        return self.data[:, idx, :]