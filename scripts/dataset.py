import torch
import numpy as np
import os
from torch.utils.data import Dataset
import sys
sys.path.append("../utility")
from franka_env import FrankaEnv
from Utility import data_collecter

class PolynomialDataCollector:
    """
    A data collection module for the toy Koopman example.
    Now parameterized by m and system coefficients.
    """
    def __init__(self, state_dim=3, m=20, a1=0.9, a2=0.9, a3=0.9, b=np.full((20-2,), 0.9)):
        self.state_dim = state_dim
        self.m = m
        self.a1 = a1 if a1 is not None else np.random.uniform(-2, 2)
        self.a2 = a2 if a2 is not None else np.random.uniform(-2, 2)
        self.a3 = a3 if a3 is not None else np.random.uniform(-2, 2)
        # b: polynomial coefficients for p=1,..., m-2
        if b is None:
            self.b = np.random.uniform(-2, 2, size=(m-2,))
        else:
            self.b = b

    def random_state(self):
        # Sample a random initial state uniformly from [-1, 1]^state_dim
        return np.random.uniform(-1, 1, size=(self.state_dim,)).astype(np.float64)
    
    def simulate_dynamics(self, x):
        """
        Simulate the nonlinear dynamics:
          x1_next = a1 * x1
          x2_next = a2 * x2
          x3_next = a3 * x3 + sum_{p=1}^{m-2} b_p * (x1)^p
        x is a numpy array of shape (batch, 3)
        """
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        poly_sum = np.zeros_like(x1)
        for p in range(1, self.m-1):  # p = 1,..., m-2
            poly_sum += self.b[p-1] * (x1 ** p)
        x1_next = self.a1 * x1
        x2_next = self.a2 * x2
        x3_next = self.a3 * x3 + poly_sum
        x_next = np.stack([x1_next, x2_next, x3_next], axis=1)
        return x_next

    def collect_koopman_data(self, traj_num, steps):
        """
        Collect trajectories for Koopman training.
        """
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
    """
    A data collection module for the infinite-dimensional (logistic map) dynamics.
    Dynamics:
       x_next = lambda_param * x * (1 - x)
    where x is a numpy array of shape (batch, 1) with values in [0, 1].
    """
    def __init__(self, state_dim=1, lambda_param=0.5):
        self.state_dim = state_dim
        if lambda_param is not None:
            self.lambda_param = lambda_param
        else:
            self.lambda_param = np.random.uniform(0, 2)
        print("lambda:", self.lambda_param)
        
    def random_state(self):
        # Sample a random initial state uniformly from [0, 1]^state_dim
        return np.random.uniform(0, 1, size=(self.state_dim,)).astype(np.float64)
    
    def simulate_dynamics(self, x):
        # Simulate the logistic map dynamics: x_next = lambda_param * x * (1 - x)
        return self.lambda_param * x * (1 - x)
    
    def collect_koopman_data(self, traj_num, steps):
        """
        Collect trajectories for Koopman training.
        """
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
    
def Obs(o):
    # Use only selected components of the observation
    return np.concatenate((o[:3], o[7:]), axis=0)

class FrankaDataCollecter():
    def __init__(self):
        self.env = FrankaEnv(render=False)
        self.Nstates = 17
        self.uval = 0.12
        self.udim = 7
        self.reset_joint_state = np.array(self.env.reset_joint_state)

    def collect_koopman_data(self, traj_num, steps):
        train_data = np.empty((steps+1, traj_num, self.Nstates + self.udim))
        for traj_i in range(traj_num):
            noise = (np.random.rand(7) - 0.5) * 2 * 0.2
            joint_init = self.reset_joint_state + noise
            joint_init = np.clip(joint_init, self.env.joint_low, self.env.joint_high)
            s0 = self.env.reset_state(joint_init)
            s0 = Obs(s0)
            u10 = (np.random.rand(7) - 0.5) * 2 * self.uval
            train_data[0, traj_i, :] = np.concatenate([u10.reshape(-1), s0.reshape(-1)], axis=0).reshape(-1)
            if traj_i % 1000 == 0:
                print("Trajectory:", traj_i)
            for i in range(1, steps+1):
                s0 = self.env.step(u10)
                s0 = Obs(s0)
                u10 = (np.random.rand(7) - 0.5) * 2 * self.uval
                train_data[i, traj_i, :] = np.concatenate([u10.reshape(-1), s0.reshape(-1)], axis=0).reshape(-1)
        return train_data
    
class G1Go2DataCollecter():
    def __init__(self, env_name, normalize=True):
        self.normalize = normalize

        go2_tracking_path_0 = '2025-03-24-20-45-16_trajnum30000_trajlen15'
        go2_tracking_path_1 = '2025-03-24-21-14-03_trajnum30000_trajlen15'
        go2_tracking_path_2 = '2025-03-24-21-57-32_trajnum30000_trajlen15'
        g1_tracking_path_0 = '2025-03-23-23-31-06_trajnum30000_trajlen15'
        g1_tracking_path_1 = '2025-03-23-23-59-32_trajnum30000_trajlen15'
        g1_tracking_path_2 = '2025-03-24-00-43-16_trajnum30000_trajlen15'

        try:
            if env_name == 'Go2':
                tracking_dataset_path_0 = f"../data/datasets/unitree_go2_flat/tracking_dataset/{go2_tracking_path_0}.npz"
                tracking_dataset_path_1 = f"../data/datasets/unitree_go2_flat/tracking_dataset/{go2_tracking_path_1}.npz"
                tracking_dataset_path_2 = f"../data/datasets/unitree_go2_flat/tracking_dataset/{go2_tracking_path_2}.npz"

            elif env_name == 'G1':
                tracking_dataset_path_0 = f"../data/datasets/g1_flat/tracking_dataset/{g1_tracking_path_0}.npz"
                tracking_dataset_path_1 = f"../data/datasets/g1_flat/tracking_dataset/{g1_tracking_path_1}.npz"
                tracking_dataset_path_2 = f"../data/datasets/g1_flat/tracking_dataset/{g1_tracking_path_2}.npz"
        except:
            raise ValueError("Dataset not found for the given environment.")
        
        self.data_pathes = [tracking_dataset_path_0, tracking_dataset_path_1, tracking_dataset_path_2]
    
    def get_data(self, data_paths, steps=15, normalize=True):
        state_data = []
        action_data = []
        for path in data_paths:
            state_data.append(np.load(path)['state_data'])
            action_data.append(np.load(path)['action_data'])
            if state_data[-1].shape[0] != steps+1:
                state_data[-1] = state_data[-1][:steps+1, :, :]
                action_data[-1] = action_data[-1][:steps, :, :]
            print(state_data[-1].shape)
            print(action_data[-1].shape)

        state_data = np.concatenate(state_data, axis=1)
        action_data = np.concatenate(action_data, axis=1)

        if normalize:
            state_data_mean = np.mean(state_data, axis=(0, 1))
            state_data_std = np.std(state_data, axis=(0, 1))
            state_data = (state_data - state_data_mean) / state_data_std

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
        return self.get_data(self.data_pathes, steps, normalize=self.normalize)[:, :traj_num, :]
    
class KoopmanDatasetCollector():
    def __init__(self, env_name, train_samples=60000, val_samples=20000, test_samples=20000, Ksteps=15, device="cuda"):
        """
        data: Tensor of shape (steps, num_trajectories, data_dim)
        """

        if env_name == "Polynomial" or env_name == "LogisticMap":
            Ksteps = 1
            
        data_path = f"../data/datasets/dataset_{env_name}_Ktrain_{train_samples}_Kval_{val_samples}_Ktest_{test_samples}_Ksteps_{Ksteps}.pt"
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
            self.state_dim = collector.Nstates
            self.u_dim = collector.udim
        elif env_name == "DoublePendulum" or env_name == "DampingPendulum":
            collector = data_collecter(env_name)
            self.state_dim = collector.Nstates
            self.u_dim = collector.udim
        elif env_name == "G1":
            collector = G1Go2DataCollecter(env_name, normalize=True)
            self.state_dim = 87
            self.u_dim = 37
        elif env_name == "Go2":
            collector = G1Go2DataCollecter(env_name, normalize=True)
            self.state_dim = 37
            self.u_dim = 12
        else:
            raise ValueError("Unknown environment name.")
            
        if not os.path.exists(data_path):
            data = collector.collect_koopman_data(train_samples+val_samples+test_samples, Ksteps)
            permutation = np.random.permutation(data.shape[1])
            shuffled = data[:, permutation, :]
            train_data = shuffled[:, :50000, :]
            val_data = shuffled[:, 50000:70000, :]
            test_data = shuffled[:, 70000:90000, :]

            torch.save({"Ktrain_data": train_data, "Kval_data": val_data, "Ktest_data": test_data}, data_path)

        dataset = torch.load(data_path, weights_only=False)
        self.train_data = dataset['Ktrain_data']
        self.val_data = dataset['Kval_data']
        self.test_data = dataset['Ktest_data']
    
    def get_data(self):
        return self.train_data, self.val_data, self.test_data
    
class KoopmanDataset(Dataset):
    def __init__(self, data):
        # data is shape (Ksteps, num_trajectories, data_dim)
        self.data = data
    def __len__(self):
        return self.data.shape[1]     # number of trajectories
    def __getitem__(self, idx):
        return self.data[:, idx, :]  # shape (Ksteps, data_dim)
