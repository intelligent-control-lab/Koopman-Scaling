import torch
import numpy as np
import torch.nn as nn
import random
import copy
import os
import itertools
import wandb
from torch.utils.data import Dataset, DataLoader
from franka_env import FrankaEnv
import math

# -------------------------------
# Franka Observation Function and Data Collection
# -------------------------------
def Obs(o):
    # Use only selected components of the observation
    return np.concatenate((o[:3], o[7:]), axis=0)

class data_collecter():
    def __init__(self, env_name) -> None:
        self.env_name = env_name
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
            for i in range(1, steps+1):
                s0 = self.env.step(u10)
                s0 = Obs(s0)
                u10 = (np.random.rand(7) - 0.5) * 2 * self.uval
                train_data[i, traj_i, :] = np.concatenate([u10.reshape(-1), s0.reshape(-1)], axis=0).reshape(-1)
        return train_data

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
    def __init__(self, encode_layers, Nkoopman, u_dim):
        super(Network, self).__init__()
        layers_list = []
        for layer_i in range(len(encode_layers)-1):
            layers_list.append(ResidualBlock(encode_layers[layer_i], encode_layers[layer_i+1]))
        self.encode_net = nn.Sequential(*layers_list)
        self.Nkoopman = Nkoopman
        self.u_dim = u_dim
        self.lA = nn.Linear(Nkoopman, Nkoopman, bias=False)
        self.lA.weight.data = gaussian_init_(Nkoopman, std=1)
        U, _, V = torch.svd(self.lA.weight.data)
        self.lA.weight.data = torch.mm(U, V.t()) * 0.9
        self.lB = nn.Linear(u_dim, Nkoopman, bias=False)

    def encode(self, x):
        # Concatenate the original input with its residual encoded output.
        return torch.cat([x, self.encode_net(x)], axis=-1)

    def forward(self, x, b):
        return self.lA(x) + self.lB(b)

# -------------------------------
# Loss Functions
# -------------------------------
def Klinear_loss(data, net, mse_loss, u_dim=1, gamma=0.99, Nstate=4):
    steps, train_traj_num, Nkoopman = data.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    X_current = net.encode(data[0, :, u_dim:])
    beta = 1.0
    beta_sum = 0.0
    Kloss = torch.zeros(1, dtype=torch.float64).to(device)
    Ploss = torch.zeros(1, dtype=torch.float64).to(device)
    for i in range(steps-1):
        X_current = net.forward(X_current, data[i, :, :u_dim])
        beta_sum += beta
        Ploss += beta * mse_loss(X_current[:, :Nstate], data[i+1, :, u_dim:])
        Y = net.encode(data[i+1, :, u_dim:])
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
# Training Function with modifications for Franka
# -------------------------------
def train(env_name, train_steps=20000, all_loss=0, encode_dim=12, layer_depth=5, c_loss=1, 
          gamma=0.99, detach=0, Ktrain_samples=50000, Ktest_samples=20000, Ksteps=15, seed=42, batch_size=64,
          cov_weight=1, initial_lr=1e-3, lr_step=1000, lr_gamma=0.95):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Use the Franka data collection pipeline
    data_collect = data_collecter(env_name)

    dataset_dir = "../Data/datasets/"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    dataset_filename = os.path.join(dataset_dir,
                    f"dataset_{env_name}_Ktrain_{Ktrain_samples}_Ksteps_{Ksteps}_seed_{seed}.pt")
    print("Dataset filename:", dataset_filename)
    if os.path.exists(dataset_filename):
        print("Loading dataset from", dataset_filename)
        dataset = torch.load(dataset_filename, weights_only=False)
        Ktrain_data = dataset['Ktrain_data']
        Ktest_data = dataset['Ktest_data']
    else:
        print("Generating new dataset...")
        Ktest_data = data_collect.collect_koopman_data(Ktest_samples, Ksteps)
        Ktrain_data = data_collect.collect_koopman_data(Ktrain_samples, Ksteps)
        dataset = {'Ktrain_data': Ktrain_data, 'Ktest_data': Ktest_data}
        torch.save(dataset, dataset_filename)
    Ktest_samples = Ktest_data.shape[1]
    print("Test data ok!, shape:", Ktest_data.shape)
    print("Train data ok!, shape:", Ktrain_data.shape)

    if isinstance(Ktrain_data, np.ndarray):
        Ktrain_data = torch.from_numpy(Ktrain_data)
    if isinstance(Ktest_data, np.ndarray):
        Ktest_data = torch.from_numpy(Ktest_data)

    in_dim = Ktest_data.shape[-1] - data_collect.udim
    Nstate = in_dim

    layers = get_layers(in_dim, encode_dim, layer_depth)
    Nkoopman = in_dim + layers[-1]
    print("Encoder layers:", layers)

    net = Network(layers, Nkoopman, data_collect.udim)
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
               })

    best_loss = 1000.0

    step = 0
    val_losses = []

    Ktrain_data = Ktrain_data.double()
    Ktest_data = Ktest_data.double()
    Ktest_data = Ktest_data.to(device)

    train_dataset = KoopmanDataset(Ktrain_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    udim = data_collect.udim

    while step < train_steps:
        for batch in train_loader:
            if step >= train_steps:
                break
            X = batch.permute(1, 0, 2).to(device)
            Kloss, Ploss = Klinear_loss(X, net, mse_loss, udim, gamma, Nstate)
            x_state = X[0, :, udim:]

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
            torch.nn.utils.clip_grad_norm_(net.lB.parameters(), max_norm)
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
                    Kloss_eval, Ploss_eval = Klinear_loss(Ktest_data, net, mse_loss, udim, gamma, Nstate)
                    x_state_eval = Ktest_data[0, :, udim:]
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
    envs = ["Franka"]

    for env, encode_dim, cov_weight, layer_depth, random_seed, c_loss in itertools.product(envs, encode_dims, cov_weights, layer_depths, random_seeds, c_losses):
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
            Ksteps=10,
            seed=random_seed,
            batch_size=64,
            cov_weight=cov_weight,
            initial_lr=1e-3,
            lr_step=1000,
            lr_gamma=0.95)

if __name__ == "__main__":
    main()
