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
import gym

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

class FrankaDataCollector(object):
    def __init__(self, render=False, ts=0.002, env_name="FrankaEnv"):
        # Environment setup
        self.frame_skip = 10
        if render:
            self.client = pb.connect(pb.GUI)
        else:
            self.client = pb.connect(pb.DIRECT)
        pb.setTimeStep(ts)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        planeID = pb.loadURDF('plane.urdf')
        self.robot = pb.loadURDF('../utility/franka_description/robots/franka_panda.urdf', [0., 0., 0.], useFixedBase=1)
        pb.setGravity(0, 0, -9.81)

        # Parameters
        self.reset_joint_state = [0., -0.78, 0., -2.35, 0., 1.57, 0.78]
        self.ee_id = 7
        self.sat_val = 0.3
        self.joint_low = np.array([-2.9, -1.8, -2.9, -3.0, -2.9, -0.08, -2.9])
        self.joint_high = np.array([2.9, 1.8, 2.9, 0.08, 2.9, 3.0, 2.9])
        self.Nstates = 17
        self.udim = 7
        self.dt = self.frame_skip * ts
        self.uval = 0.12
        self.env_name = env_name

        # Initialize state
        self.reset()

    def reset(self):
        for i, jnt in enumerate(self.reset_joint_state):
            pb.resetJointState(self.robot, i, self.reset_joint_state[i])
        return self.get_state()

    def reset_state(self, joint):
        for i, jnt in enumerate(joint):
            pb.resetJointState(self.robot, i, joint[i])
        return self.get_state()

    def step(self, action):
        a = np.clip(action, -self.sat_val, self.sat_val)
        pb.setJointMotorControlArray(
            self.robot, range(7),
            pb.VELOCITY_CONTROL, targetVelocities=a)
        for _ in range(self.frame_skip):
            pb.stepSimulation()
        return self.get_state()

    def get_ik(self, position, orientation=None):
        if orientation is None:
            jnts = pb.calculateInverseKinematics(self.robot, self.ee_id, position)[:7]
        else:
            jnts = pb.calculateInverseKinematics(self.robot, self.ee_id, position, orientation)[:7]
        return jnts

    def get_state(self):
        jnt_st = pb.getJointStates(self.robot, range(7))
        ee_state = pb.getLinkState(self.robot, self.ee_id)[-2:]  # position, orientation
        jnt_ang = []
        jnt_vel = []
        for jnt in jnt_st:
            jnt_ang.append(jnt[0])
            jnt_vel.append(jnt[1])
        self.state = np.concatenate([ee_state[0], ee_state[1], jnt_ang, jnt_vel])
        return self.state.copy()

    def Obs(self, o):
        return np.concatenate((o[:3], o[7:]), axis=0)

    def collect_koopman_data(self, traj_num, steps):
        train_data = np.empty((steps + 1, traj_num, self.Nstates + self.udim))
        for traj_i in tqdm(range(traj_num)):
            noise = (np.random.rand(7) - 0.5) * 2 * 0.2
            joint_init = np.clip(np.array(self.reset_joint_state) + noise, self.joint_low, self.joint_high)
            s0 = self.reset_state(joint_init)
            s0 = self.Obs(s0)
            u10 = (np.random.rand(7) - 0.5) * 2 * self.uval
            train_data[0, traj_i, :] = np.concatenate([u10.reshape(-1), s0.reshape(-1)], axis=0).reshape(-1)
            for i in range(1, steps + 1):
                s0 = self.step(u10)
                s0 = self.Obs(s0)
                u10 = (np.random.rand(7) - 0.5) * 2 * self.uval
                train_data[i, traj_i, :] = np.concatenate([u10.reshape(-1), s0.reshape(-1)], axis=0).reshape(-1)
        return train_data

class G1Go2DataCollector():
    def __init__(self, env_name, use_initial_data=False):
        if use_initial_data:
            g1_initial_path = 'None_trajnum90000_trajlen100'
            go2_initial_path = 'None_trajnum89947_trajlen100'
            try:
                if env_name == 'Go2':
                    initial_dataset_path = f"../data/unitree_go2_flat/initial_dataset/{go2_initial_path}.npz"
                elif env_name == 'G1':
                    initial_dataset_path = f"../data/g1_flat/initial_dataset/{g1_initial_path}.npz"
            except:
                raise ValueError("Dataset not found for the given environment.")
            self.data_pathes = [initial_dataset_path]
        else:
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
            try:
                if env_name == 'Go2':
                    tracking_dataset_path_0 = f"../data/unitree_go2_flat/tracking_dataset/{go2_tracking_path_0}.npz"
                    tracking_dataset_path_1 = f"../data/unitree_go2_flat/tracking_dataset/{go2_tracking_path_1}.npz"
                    tracking_dataset_path_2 = f"../data/unitree_go2_flat/tracking_dataset/{go2_tracking_path_2}.npz"
                    tracking_dataset_path_3 = f"../data/unitree_go2_flat/tracking_dataset/{go2_tracking_path_3}.npz"
                    self.data_pathes = [tracking_dataset_path_0, tracking_dataset_path_1, tracking_dataset_path_2, tracking_dataset_path_3]
                elif env_name == 'G1':
                    tracking_dataset_path_0 = f"../data/g1_flat/tracking_dataset/{g1_tracking_path_0}.npz"
                    tracking_dataset_path_1 = f"../data/g1_flat/tracking_dataset/{g1_tracking_path_1}.npz"
                    tracking_dataset_path_2 = f"../data/g1_flat/tracking_dataset/{g1_tracking_path_2}.npz"
                    tracking_dataset_path_3 = f"../data/g1_flat/tracking_dataset/{g1_tracking_path_3}.npz"
                    tracking_dataset_path_4 = f"../data/g1_flat/tracking_dataset/{g1_tracking_path_4}.npz"
                    tracking_dataset_path_5 = f"../data/g1_flat/tracking_dataset/{g1_tracking_path_5}.npz"
                    self.data_pathes = [tracking_dataset_path_0, tracking_dataset_path_1, tracking_dataset_path_2, tracking_dataset_path_3, tracking_dataset_path_4, tracking_dataset_path_5]
            except:
                raise ValueError("Dataset not found for the given environment.")
    
    def get_data(self, data_paths, steps=15):
        state_data = []
        action_data = []
        for path in data_paths:
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
        return self.get_data(self.data_pathes, steps)[:, :traj_num, :]
    
class KinovaDataCollector():
    def __init__(self):
        self.state_dim = 14
        self.u_dim = 7
        self.data_pathes = ['output_20250402_172619.txt',
                            'output_20250402_182836.txt',
                            'output_20250402_195709.txt',
                            'output_20250402_205831.txt',
                            'output_20250403_104412.txt']
    
    def get_data(self, data_paths, steps=10):
        def process_data(file_path):
            df = pd.read_csv(f'../data/kinova_data/{file_path}', 
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
        return self.get_data(self.data_pathes, steps)[:, :traj_num, :]

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

        norm_str = "norm" if self.normalize else "nonorm"
        if env_name == "Polynomial":
            data_path = f"../data/datasets/dataset_{env_name}_{norm_str}_m_{m}_Ktrain_{train_samples}_Kval_{val_samples}_Ktest_{test_samples}_Ksteps_{Ksteps}.pt"
        else:
            data_path = f"../data/datasets/dataset_{env_name}_{norm_str}_Ktrain_{train_samples}_Kval_{val_samples}_Ktest_{test_samples}_Ksteps_{Ksteps}.pt"
        
        self.u_dim = None
        self.state_dim = None

        if env_name == "Polynomial":
            collector = PolynomialDataCollector(m=m)
            self.state_dim = collector.state_dim
        elif env_name == "LogisticMap":
            collector = LogisticMapDataCollector()
            self.state_dim = collector.state_dim
        elif env_name == "Franka":
            collector = FrankaDataCollector()
            self.state_dim = 17
            self.u_dim = 7
        elif env_name == "DoublePendulum":
            collector = DoublePendulumDataCollector()
            self.state_dim = collector.state_dim
            self.u_dim = collector.u_dim
        elif env_name == "DampingPendulum":
            collector = DampingPendulumDataCollector()
            self.state_dim = collector.state_dim
            self.u_dim = collector.u_dim
        elif env_name == "G1":
            collector = G1Go2DataCollector(env_name, use_initial_data=False)
            self.full_state_dim = 87  # Original state dimension
            self.full_u_dim = 37      # Original action dimension
            self.state_dim = 46       # Trimmed state dimension (23+23+1+6-6 = 47? Let's calculate properly)
            # G1 trimmed: 23 joint_pos + 23 joint_vel + 1 height + 6 root_state = 53
            self.state_dim = 53
            self.u_dim = 23           # Trimmed action dimension
        elif env_name == "Go2":
            collector = G1Go2DataCollector(env_name, use_initial_data=False)
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