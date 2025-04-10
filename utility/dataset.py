import torch
import numpy as np
import os
from torch.utils.data import Dataset
import pybullet as pb
import pybullet_data

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
    def __init__(self, state_dim=1, lambda_param=2):
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

class DoublePendulumDataCollector:
    def __init__(self, dt=0.05, m1=1.0, m2=1.0, L1=1.0, L2=1.0, g=9.81):
        self.dt = dt
        self.m1 = m1
        self.m2 = m2
        self.L1 = L1
        self.L2 = L2
        self.g = g
        self.state_dim = 4
        self.u_dim = 2

    def random_state(self):
        theta1 = np.random.uniform(-np.pi, np.pi)
        theta2 = np.random.uniform(-np.pi, np.pi)
        omega1 = np.random.uniform(-1, 1)
        omega2 = np.random.uniform(-1, 1)
        return np.array([theta1, theta2, omega1, omega2], dtype=np.float64)

    def random_control(self):
        u1 = np.random.uniform(-1, 1)
        u2 = np.random.uniform(-1, 1)
        return np.array([u1, u2], dtype=np.float64)

    def derivatives(self, state, u):
        theta1, theta2, omega1, omega2 = state
        u1, u2 = u
        m1, m2, L1, L2, g = self.m1, self.m2, self.L1, self.L2, self.g
        delta = theta2 - theta1

        denom = 2 * m1 + m2 - m2 * np.cos(2 * delta)
        if np.abs(denom) < 1e-6:
            denom = 1e-6

        dtheta1 = omega1
        dtheta2 = omega2

        domega1_no_control = (
            -g*(2*m1+m2)*np.sin(theta1)
            - m2*g*np.sin(theta1-2*theta2)
            - 2*np.sin(delta)*m2*(L2*omega2**2 + L1*omega1**2*np.cos(delta))
        ) / (L1 * denom)

        domega2_no_control = (
            2*np.sin(delta)*(
                L1*omega1**2*(m1+m2)
                + g*(m1+m2)*np.cos(theta1)
                + L2*omega2**2*m2*np.cos(delta)
            )
        ) / (L2 * denom)

        domega1 = domega1_no_control + u1
        domega2 = domega2_no_control + u2

        return np.array([dtheta1, dtheta2, domega1, domega2], dtype=np.float64)

    def simulate_dynamics(self, state, u):
        deriv = self.derivatives(state, u)
        next_state = state + self.dt * deriv
        return next_state

    def collect_koopman_data(self, traj_num, steps):
        data = np.empty((steps + 1, traj_num, self.state_dim + self.u_dim), dtype=np.float64)
        for traj in range(traj_num):
            state = self.random_state()
            control = self.random_control()
            data[0, traj, :] = np.concatenate([control, state])
            for i in range(1, steps + 1):
                control = self.random_control()
                state = self.simulate_dynamics(state, control)
                data[i, traj, :] = np.concatenate([control, state])
        return data


class DampingPendulumDataCollector:
    def __init__(self, dt=0.05, L=1.0, g=9.81, damping=0.5):
        self.dt = dt
        self.L = L
        self.g = g
        self.damping = damping
        self.state_dim = 2
        self.u_dim = 1

    def random_state(self):
        theta = np.random.uniform(-np.pi, np.pi)
        omega = np.random.uniform(-1, 1)
        return np.array([theta, omega], dtype=np.float64)

    def random_control(self):
        u = np.random.uniform(-1, 1)
        return np.array([u], dtype=np.float64)

    def derivatives(self, state, u):
        theta, omega = state
        control = u[0]
        dtheta = omega
        domega = - (self.g / self.L) * np.sin(theta) - self.damping * omega + control
        return np.array([dtheta, domega], dtype=np.float64)

    def simulate_dynamics(self, state, u):
        deriv = self.derivatives(state, u)
        next_state = state + self.dt * deriv
        return next_state

    def collect_koopman_data(self, traj_num, steps):
        data = np.empty((steps + 1, traj_num, self.state_dim + self.u_dim), dtype=np.float64)
        for traj in range(traj_num):
            state = self.random_state()
            control = self.random_control()
            data[0, traj, :] = np.concatenate([control, state])
            for i in range(1, steps + 1):
                control = self.random_control()
                state = self.simulate_dynamics(state, control)
                data[i, traj, :] = np.concatenate([control, state])
        return data
    
def Obs(o):
    return np.concatenate((o[:3], o[7:]), axis=0)

class FrankaDataCollecter:
    def __init__(self, render=False, ts=0.002):
        self.render = render
        if self.render:
            self.client = pb.connect(pb.GUI)
        else:
            self.client = pb.connect(pb.DIRECT)

        self.ts = ts
        self.frame_skip = 10
        pb.setTimeStep(self.ts)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())

        pb.loadURDF('plane.urdf')

        base_path = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(base_path, "franka_description/robots/franka_panda.urdf")
        self.robot = pb.loadURDF(urdf_path, [0., 0., 0.], useFixedBase=1)
        
        pb.setGravity(0, 0, -9.81)

        self.reset_joint_state = [0., -0.78, 0., -2.35, 0., 1.57, 0.78]
        self.joint_low = np.array([-2.9, -1.8, -2.9, -3.0, -2.9, -0.08, -2.9])
        self.joint_high = np.array([2.9, 1.8, 2.9, 0.08, 2.9, 3.0, 2.9])
        self.sat_val = 0.3
        
        self.state_dim = 17
        self.u_dim = 7
        self.uval = 0.12
        self.dt = self.frame_skip * self.ts

        self.init_joint_noise_scale = 0.3

        self.reset()

    def get_state(self):
        joint_states = pb.getJointStates(self.robot, list(range(7)))
        ee_state = pb.getLinkState(self.robot, 7)[-2:]
        jnt_angles = [state[0] for state in joint_states]
        jnt_velocities = [state[1] for state in joint_states]
        full_state = np.concatenate([ee_state[0], ee_state[1], jnt_angles, jnt_velocities])
        return full_state.copy()

    def reset(self):
        for i, jnt in enumerate(self.reset_joint_state):
            pb.resetJointState(self.robot, i, jnt)
        return self.get_state()

    def reset_state(self, joint_state):
        for i, jnt in enumerate(joint_state):
            pb.resetJointState(self.robot, i, jnt)
        return self.get_state()

    def step(self, action):
        a = np.clip(action, -self.sat_val, self.sat_val)
        pb.setJointMotorControlArray(
            self.robot, list(range(7)),
            controlMode=pb.VELOCITY_CONTROL, targetVelocities=a)
        for _ in range(self.frame_skip):
            pb.stepSimulation()
        return self.get_state()

    def generate_random_control(self):
        scale = np.random.uniform(0.8, 1.2)
        return (np.random.rand(7) - 0.5) * 2 * (self.uval * scale)

    def collect_koopman_data(self, traj_num, steps):
        data = np.empty((steps + 1, traj_num, self.u_dim + self.state_dim), dtype=np.float64)
        for traj in range(traj_num):
            noise = (np.random.rand(7) - 0.5) * 2 * self.init_joint_noise_scale
            joint_init = np.array(self.reset_joint_state) + noise
            joint_init = np.clip(joint_init, self.joint_low, self.joint_high)
            s0 = self.reset_state(joint_init)
            s0_obs = Obs(s0)
            u = self.generate_random_control()
            data[0, traj, :] = np.concatenate([u.reshape(-1), s0_obs.reshape(-1)])
            if traj % 1000 == 0:
                print("Trajectory:", traj)
            for t in range(1, steps + 1):
                s0 = self.step(u)
                s0_obs = Obs(s0)
                u = self.generate_random_control()
                data[t, traj, :] = np.concatenate([u.reshape(-1), s0_obs.reshape(-1)])
        return data


class G1Go2DataCollecter():
    def __init__(self, env_name, use_initial_data=False):
        if use_initial_data:
            g1_initial_path = 'None_trajnum90000_trajlen100'
            go2_initial_path = 'None_trajnum89947_trajlen100'
            try:
                if env_name == 'Go2':
                    initial_dataset_path = f"../data/datasets/unitree_go2_flat/initial_dataset/{go2_initial_path}.npz"
                elif env_name == 'G1':
                    initial_dataset_path = f"../data/datasets/g1_flat/initial_dataset/{g1_initial_path}.npz"
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
                    tracking_dataset_path_0 = f"../data/datasets/unitree_go2_flat/tracking_dataset/{go2_tracking_path_0}.npz"
                    tracking_dataset_path_1 = f"../data/datasets/unitree_go2_flat/tracking_dataset/{go2_tracking_path_1}.npz"
                    tracking_dataset_path_2 = f"../data/datasets/unitree_go2_flat/tracking_dataset/{go2_tracking_path_2}.npz"
                    tracking_dataset_path_3 = f"../data/datasets/unitree_go2_flat/tracking_dataset/{go2_tracking_path_3}.npz"
                    self.data_pathes = [tracking_dataset_path_0, tracking_dataset_path_1, tracking_dataset_path_2, tracking_dataset_path_3]
                elif env_name == 'G1':
                    tracking_dataset_path_0 = f"../data/datasets/g1_flat/tracking_dataset/{g1_tracking_path_0}.npz"
                    tracking_dataset_path_1 = f"../data/datasets/g1_flat/tracking_dataset/{g1_tracking_path_1}.npz"
                    tracking_dataset_path_2 = f"../data/datasets/g1_flat/tracking_dataset/{g1_tracking_path_2}.npz"
                    tracking_dataset_path_3 = f"../data/datasets/g1_flat/tracking_dataset/{g1_tracking_path_3}.npz"
                    tracking_dataset_path_4 = f"../data/datasets/g1_flat/tracking_dataset/{g1_tracking_path_4}.npz"
                    tracking_dataset_path_5 = f"../data/datasets/g1_flat/tracking_dataset/{g1_tracking_path_5}.npz"
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

class KoopmanDatasetCollector():
    def __init__(self, env_name, train_samples=60000, val_samples=20000, test_samples=20000, Ksteps=15, normalize=True):
        self.normalize = normalize

        norm_str = "norm" if self.normalize else "nonorm"
        data_path = f"../data/datasets/dataset_{env_name}_{norm_str}_Ktrain_{train_samples}_Kval_{val_samples}_Ktest_{test_samples}_Ksteps_{Ksteps}.pt"
        
        self.u_dim = None
        self.state_dim = None

        if env_name == "Polynomial":
            collector = PolynomialDataCollector()
            self.state_dim = collector.state_dim
        elif env_name == "LogisticMap":
            collector = LogisticMapDataCollector()
            self.state_dim = collector.state_dim
        elif env_name == "Franka":
            collector = FrankaDataCollecter()
            self.state_dim = collector.state_dim
            self.u_dim = collector.u_dim
        elif env_name == "DoublePendulum":
            collector = DoublePendulumDataCollector()
            self.state_dim = collector.state_dim
            self.u_dim = collector.u_dim
        elif env_name == "DampingPendulum":
            collector = DampingPendulumDataCollector()
            self.state_dim = collector.state_dim
            self.u_dim = collector.u_dim
        elif env_name == "G1":
            collector = G1Go2DataCollecter(env_name, use_initial_data=False)
            self.state_dim = 87
            self.u_dim = 37
        elif env_name == "Go2":
            collector = G1Go2DataCollecter(env_name, use_initial_data=False)
            self.state_dim = 37
            self.u_dim = 12
        else:
            raise ValueError("Unknown environment name.")
        
        if not os.path.exists(data_path):
            data = collector.collect_koopman_data(train_samples+val_samples+test_samples, Ksteps)
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
            
            torch.save({"Ktrain_data": train_data, "Kval_data": val_data, "Ktest_data": test_data}, data_path)

        self.train_data = torch.load(data_path, weights_only=False)["Ktrain_data"]
        self.val_data = torch.load(data_path, weights_only=False)["Kval_data"]
        self.test_data = torch.load(data_path, weights_only=False)["Ktest_data"]

    
    def get_data(self):
        return self.train_data, self.val_data, self.test_data

class KoopmanDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        return self.data[:, idx, :]