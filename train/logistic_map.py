import torch
import torch.nn as nn
import numpy as np
import random
import wandb
import os
from torch.utils.data import Dataset, DataLoader
import itertools
import copy
import math

state_dim = 1

class DataCollectorInfinite:
    """
    A data collection module for the infinite-dimensional (logistic map) dynamics.
    Dynamics:
       x_next = lambda_param * x * (1 - x)
    where x is a numpy array of shape (batch, 1) with values in [0, 1].
    """
    def __init__(self, state_dim=1, lambda_param=None):
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

# -------------------------------
# Helper functions and initializers
# -------------------------------
def get_layers(input_dim, target_dim, layer_depth=5):
    if layer_depth < 2:
        raise ValueError("Layer depth must be at least 2 (input and target layers).")

    layers = [input_dim]
    
    if input_dim < target_dim:  # Exponential increase
        factor = (target_dim / input_dim) ** (1 / (layer_depth - 1))
        for i in range(1, layer_depth - 1):
            layers.append(int(math.ceil(layers[-1] * factor)))
    else:  # Exponential decrease
        factor = (target_dim / input_dim) ** (1 / (layer_depth - 1))
        for i in range(1, layer_depth - 1):
            layers.append(int(math.ceil(layers[-1] * factor)))

    layers.append(target_dim)  # Ensure last layer is exactly target_dim
    return layers

def gaussian_init_(n_units, std=1):    
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]  
    return Omega

# -------------------------------
# Residual Block for the encoder network
# -------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        # If dimensions differ, use a linear projection for the shortcut.
        if in_features != out_features:
            self.residual = nn.Linear(in_features, out_features)
        else:
            self.residual = None

    def forward(self, x):
        out = self.linear(x)
        out = self.relu(out)
        if self.residual is not None:
            res = self.residual(x)
        else:
            res = x
        return out + res

# -------------------------------
# Custom Dataset for Koopman trajectories
# -------------------------------
class KoopmanDataset(Dataset):
    def __init__(self, data):
        """
        data: Tensor of shape (steps, num_trajectories, data_dim)
        """
        self.data = data

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        return self.data[:, idx, :]
    
# -------------------------------
# Define the Network with Residual Connections in the Encoder
# -------------------------------
class Network(nn.Module):
    def __init__(self, encode_layers, Nkoopman):
        super(Network, self).__init__()
        layers_list = []
        for layer_i in range(len(encode_layers)-1):
            layers_list.append(ResidualBlock(encode_layers[layer_i], encode_layers[layer_i+1]))
        self.encode_net = nn.Sequential(*layers_list)

        self.Nkoopman = Nkoopman
        self.lA = nn.Linear(Nkoopman, Nkoopman, bias=False)
        self.lA.weight.data = gaussian_init_(Nkoopman, std=1)
        U, _, V = torch.svd(self.lA.weight.data)
        self.lA.weight.data = torch.mm(U, V.t()) * 0.9

    def encode_only(self, x):
        return self.encode_net(x)

    def encode(self, x):
        # Combine original state x and encoded version
        return torch.cat([x, self.encode_net(x)], axis=-1)
    
    def forward(self, x):
        return self.lA(x)
    
# -------------------------------
# Loss Functions
# -------------------------------
def Klinear_loss(data, net, mse_loss, gamma=0.99):
    steps, train_traj_num, Nkoopman = data.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    X_current = net.encode(data[0, :])
    beta = 1.0
    beta_sum = 0.0
    Kloss = torch.zeros(1, dtype=torch.float64).to(device)
    Ploss = torch.zeros(1, dtype=torch.float64).to(device)
    for i in range(steps-1):
        X_current = net.forward(X_current)
        beta_sum += beta
        Ploss += beta * mse_loss(X_current[:, :state_dim], data[i+1, :])
        Y = net.encode(data[i+1, :])
        Kloss += beta * mse_loss(X_current, Y)
        beta *= gamma
    Kloss = Kloss / beta_sum
    Ploss = Ploss / beta_sum
    return Kloss, Ploss

def Cov_loss(net, x):
    z = net.encode(x)
    z_mean = torch.mean(z, dim=0, keepdim=True)
    z_centered = z - z_mean
    cov_matrix = (z_centered.t() @ z_centered) / (z_centered.size(0) - 1)
    diag_cov = torch.diag(torch.diag(cov_matrix))
    off_diag = cov_matrix - diag_cov
    loss = torch.norm(off_diag, p='fro')**2
    return loss

# -------------------------------
# Training Function
# -------------------------------
def train(env_name, train_steps=20000, all_loss=0, encode_dim=12, layer_depth=5, c_loss=1, 
            gamma=0.99, Ktrain_samples=50000, Ktest_samples=20000, Ksteps=1, seed=42, batch_size=64,
          cov_weight=1, initial_lr=1e-3, lr_step=1000, lr_gamma=0.95, lambda_param=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset_dir = "../Data/datasets/"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    dataset_filename = os.path.join(dataset_dir,
                    f"dataset_{env_name}_Ktrain_{Ktrain_samples}_Ksteps_{Ksteps}_seed_{seed}_lambda_{lambda_param}.npz")
    if os.path.exists(dataset_filename):
        print(f"Loading dataset from {dataset_filename}")
        data_npz = np.load(dataset_filename)
        Ktrain_data = data_npz["Ktrain_data"]
        Ktest_data = data_npz["Ktest_data"]
    else:
        print(f"Generating dataset...")
        # Use fixed dynamics: a1, a2, a3 = a_val; b filled with a_val
        data_collect = DataCollectorInfinite(state_dim=state_dim, lambda_param=lambda_param)
        Ksteps = 2
        total_samples = Ktrain_samples + Ktest_samples
        koopman_data = data_collect.collect_koopman_data(total_samples, Ksteps)
        indices = np.arange(total_samples)
        np.random.shuffle(indices)
        train_idx = indices[:Ktrain_samples]
        test_idx = indices[Ktrain_samples:]
        Ktrain_data = koopman_data[:, train_idx, :]
        Ktest_data = koopman_data[:, test_idx, :]
        # Save dataset locally for reuse
        np.savez_compressed(dataset_filename, Ktrain_data=Ktrain_data, Ktest_data=Ktest_data)
        print(f"Dataset saved to {dataset_filename}")


    print("Test data ok!, shape:", Ktest_data.shape)
    print("Train data ok!, shape:", Ktrain_data.shape)

    if isinstance(Ktrain_data, np.ndarray):
        Ktrain_data = torch.from_numpy(Ktrain_data)
    if isinstance(Ktest_data, np.ndarray):
        Ktest_data = torch.from_numpy(Ktest_data)

    layers = get_layers(state_dim, encode_dim, layer_depth)
    Nkoopman = state_dim + layers[-1]
    print("Encoder layers:", layers)

    net = Network(layers, Nkoopman)
    eval_step = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.double()
    mse_loss = nn.MSELoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    
    for name, param in net.named_parameters():
        print("model:", name, param.requires_grad)

    wandb.init(project="Results_0", 
               name=f"{env_name}_edim{encode_dim}_closs{'on' if c_loss else 'off'}_seed{seed}",
               config={
                    "env_name": env_name,
                    "train_steps": train_steps,
                    "all_loss": all_loss,
                    "encode_dim": encode_dim,
                    "layer_depth": layer_depth,
                    "c_loss": c_loss,
                    "gamma": gamma,
                    "Ktrain_samples": Ktrain_data.shape[1],
                    "Ktest_samples": Ktest_data.shape[1],
                    "Ksteps": Ktrain_data.shape[0],
                    "seed": seed,
                    "initial_lr": initial_lr,
                    "lr_step": lr_step,
                    "lr_gamma": lr_gamma,
                    "cov_weight": cov_weight,
                    "batch_size": batch_size,
                    "lambda_param": lambda_param
               })

    best_loss = 1000.0

    step = 0
    val_losses = []

    Ktrain_data = Ktrain_data.double()
    Ktest_data = Ktest_data.double()
    Ktest_data = Ktest_data.to(device)

    train_dataset = KoopmanDataset(Ktrain_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    while step < train_steps:
        for batch in train_loader:
            if step >= train_steps:
                break
            X = batch.permute(1, 0, 2).to(device)
            Kloss, Ploss = Klinear_loss(X, net, mse_loss, gamma)
            x_state = X[0, :]

            Closs = Cov_loss(net, x_state)
            factor = cov_weight / (encode_dim ** 0.5)
            if Kloss < 0.001:
                factor = factor * Kloss.item() * 1000

            if all_loss:
                loss = Kloss + factor * Closs if c_loss else Kloss
            else:
                loss = Ploss + factor * Closs if c_loss else Ploss
            optimizer.zero_grad()
            loss.backward()
            max_norm = 1.0  # Change this value based on needs
            torch.nn.utils.clip_grad_norm_(net.lA.parameters(), max_norm)
            optimizer.step()
            scheduler.step()

            wandb.log({
                "Train/Kloss": Kloss.item() if torch.is_tensor(Kloss) else Kloss,
                "Train/Ploss": Ploss.item() if torch.is_tensor(Ploss) else Ploss,
                "Train/CovLoss": Closs.item() if torch.is_tensor(Closs) else Closs,
                "step": step
            })

            if step % eval_step == 0:
                with torch.no_grad():
                    Kloss_eval, Ploss_eval = Klinear_loss(Ktest_data, net, mse_loss, gamma)
                    x_state_eval = Ktest_data[0, :]
                    Closs_eval = Cov_loss(net, x_state_eval)
                    loss_eval = Ploss_eval

                    val_losses.append(loss_eval.item())

                    if loss_eval < best_loss:
                        best_loss = copy.copy(Ploss_eval)
                        best_state_dict = copy.copy(net.state_dict())
                        saved_dict = {'model':best_state_dict,'layer':layers}
                        torch.save(saved_dict, f"../Data/models/best_model_{env_name}_{encode_dim}_{c_loss}_{seed}.pth")

                    wandb.log({
                        "Eval/Kloss": Kloss_eval.item(),
                        "Eval/Ploss": Ploss_eval.item(),
                        "Eval/CovLoss": Closs_eval.item(),
                        "Eval/best_loss": best_loss.item() if torch.is_tensor(best_loss) else best_loss,
                        "step": step,
                    })
                    print("Step:{} Prediction loss:{}".format(step, Ploss_eval.item()))

            step += 1

    if len(val_losses) >= 10:
        convergence_loss = np.mean(val_losses[-10:])
    else:
        convergence_loss = np.mean(val_losses) if len(val_losses) > 0 else None

    print("END - Best loss: {}  Convergence loss: {}".format(best_loss, convergence_loss))
    wandb.log({"best_loss": best_loss, "convergence_loss": convergence_loss})
    wandb.finish()

def main():
    c_losses = [0, 1]
    encode_dims = [1, 4, 16, 64, 256, 512, 1024]
    cov_weights = [1]
    layer_depths = [5]
    random_seeds = [1, 2, 3]
    envs = ["LogisticMap"]
    lambda_params = [0.5]

    for env, lambda_param, encode_dim, cov_weight, layer_depth, random_seed, c_loss in itertools.product(envs, lambda_params, encode_dims, cov_weights, layer_depths, random_seeds, c_losses):
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

        train(env_name=env,
            train_steps=100000,
            all_loss=1,
            encode_dim=encode_dim,
            layer_depth=layer_depth,
            c_loss=c_loss,
            gamma=0.8,
            Ktrain_samples=50000,
            Ktest_samples=20000, 
            Ksteps=1,
            seed=random_seed,
            batch_size=64,
            cov_weight=cov_weight,
            initial_lr=1e-3,
            lr_step=1000,
            lr_gamma=0.95,
            lambda_param=lambda_param)

if __name__ == "__main__":
    main()