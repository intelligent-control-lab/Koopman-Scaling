import torch
import numpy as np
import torch.nn as nn
import random
import copy
import itertools
import wandb
from torch.utils.data import DataLoader
import math
import os
from dataset import KoopmanDatasetCollector, KoopmanDataset
from network import KoopmanNet

def get_layers(input_dim, target_dim, layer_depth=5):
    if layer_depth < 2:
        raise ValueError("Layer depth must be at least 2 (input and target layers).")
    
    layers = []
    for i in range(layer_depth):
        progress = i / (layer_depth - 1)
        val = input_dim * (target_dim / input_dim) ** progress
        
        if input_dim < target_dim:
            val = min(val, target_dim)
            layer_val = int(math.ceil(val))
        else:
            val = max(val, target_dim)
            layer_val = int(math.floor(val))
        
        layers.append(layer_val)

    layers[0] = input_dim
    layers[-1] = target_dim
    return layers

def Klinear_loss(data, net, mse_loss, u_dim, gamma, device):
    if u_dim is None:
        steps, traj_num, state_dim = data.shape
        X_current = net.encode(data[0, :])
        initial_encoding = X_current
        beta = 1.0
        beta_sum = 0.0
        loss = torch.zeros(1, dtype=torch.float64).to(device)
        for i in range(steps-1):
            X_current = net.forward(X_current, None)
            beta_sum += beta
            loss += beta * mse_loss(X_current[:, :state_dim], data[i+1, :])
            beta *= gamma
        return loss / beta_sum, initial_encoding
    
    steps, traj_num, N = data.shape
    state_dim = N - u_dim
    X_current = net.encode(data[0, :, u_dim:])
    initial_encoding = X_current
    beta = 1.0
    beta_sum = 0.0
    loss = torch.zeros(1, dtype=torch.float64).to(device)
    for i in range(steps-1):
        X_current = net.forward(X_current, data[i, :, :u_dim])
        beta_sum += beta
        loss += beta * mse_loss(X_current[:, :state_dim], data[i+1, :, u_dim:])
        beta *= gamma
    return loss / beta_sum, initial_encoding

def cov_loss(z):
    z_mean = torch.mean(z, dim=0, keepdim=True)
    z_centered = z - z_mean
    cov_matrix = (z_centered.t() @ z_centered) / (z_centered.size(0) - 1)
    diag_cov = torch.diag(torch.diag(cov_matrix))
    off_diag = cov_matrix - diag_cov
    return torch.norm(off_diag, p='fro')**2

def train(project_name, env_name, train_samples=60000, val_samples=20000, test_samples=20000, Ksteps=15,
          train_steps=20000, encode_dim=16, layer_depth=None, cov_reg=0, gamma=0.99, seed=42, batch_size=64, 
          initial_lr=1e-3, lr_step=1000, lr_gamma=0.95, val_step=1000, max_norm=1, cov_reg_weight=1, normalize=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    norm_str = "norm" if normalize else "nonorm"

    if not os.path.exists(f"../log/best_models/{project_name}/"):
        os.makedirs(f"../log/best_models/{project_name}/")

    print("Loading dataset...")

    data_collector = KoopmanDatasetCollector(env_name, train_samples, val_samples, test_samples, Ksteps, device, normalize=normalize)
    Ktrain_data, Kval_data, Ktest_data = data_collector.get_data()

    Ktrain_data = torch.from_numpy(Ktrain_data).double()
    Kval_data = torch.from_numpy(Kval_data).double()

    u_dim = data_collector.u_dim
    state_dim = data_collector.state_dim

    print("u_dim:", u_dim)
    print("state_dim:", state_dim)

    print("Train data shape:", Ktrain_data.shape)
    print("Validation data shape:", Kval_data.shape)
    print("Test data shape:", Ktest_data.shape)

    if layer_depth is None:
        layer_depth = math.floor(np.log2(1024/state_dim)) + 1

    layers = get_layers(state_dim, encode_dim, layer_depth)
    Nkoopman = state_dim + layers[-1]

    print("Encoder layers:", layers)

    net = KoopmanNet(layers, Nkoopman, u_dim)
    net.to(device)
    net.double()
    mse_loss = nn.MSELoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=initial_lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    
    for name, param in net.named_parameters():
        print("model:", name, param.requires_grad)

    wandb.init(project=project_name, 
               name=f"{env_name}_edim{encode_dim}_closs{'on' if cov_reg else 'off'}_seed{seed}",
               config={
                    "env_name": env_name,
                    "train_steps": train_steps,
                    "encode_dim": encode_dim,
                    "layer_depth": layer_depth,
                    "c_loss": cov_reg,
                    "gamma": gamma,
                    "Ktrain_samples": Ktrain_data.shape[1],
                    "Kval_samples": Kval_data.shape[1],
                    "Ktest_samples": Ktest_data.shape[1],
                    "Ksteps": Ktrain_data.shape[0],
                    "seed": seed,
                    "initial_lr": initial_lr,
                    "lr_step": lr_step,
                    "lr_gamma": lr_gamma,
                    "batch_size": batch_size,
                    "max_norm": max_norm,
                    "cov_reg_weight": cov_reg_weight,
               })

    best_loss = 1e10
    step = 0
    val_losses = []

    train_dataset = KoopmanDataset(Ktrain_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    Kval_data = Kval_data.to(device)

    while step < train_steps:
        for batch in train_loader:
            if step >= train_steps:
                break
            X = batch.permute(1, 0, 2).to(device)

            Kloss, initial_encoding = Klinear_loss(X, net, mse_loss, u_dim, gamma, device)

            Closs = cov_loss(initial_encoding)

            if cov_reg:
                factor = initial_encoding.shape[1] * (initial_encoding.shape[1] - 1)
                loss = Kloss + cov_reg_weight * Closs / factor
            else:
                loss = Kloss


            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm)

            optimizer.step()
            scheduler.step()

            wandb.log({
                "Train/Kloss": Kloss.item(),
                "Train/CovLoss": Closs.item(),
                "step": step
            })

            if step % val_step == 0:
                with torch.no_grad():
                    Kloss_val, initial_encoding = Klinear_loss(Kval_data, net, mse_loss, u_dim, gamma, device)
                    Closs_val = cov_loss(initial_encoding)

                    val_losses.append(Kloss_val.item())
                    
                    if Kloss_val < best_loss:
                        best_loss = copy.copy(Kloss_val)
                        best_state_dict = copy.copy(net.state_dict())
                        saved_dict = {'model':best_state_dict,'layer':layers}
                        torch.save(saved_dict, f"../log/best_models/{project_name}/best_model_{norm_str}_{env_name}_{encode_dim}_{cov_reg}_{seed}.pth")

                    wandb.log({
                        "Val/Kloss": Kloss_val.item(),
                        "Val/CovLoss": Closs_val.item(),
                        "Val/best_Kloss": best_loss.item(),
                        "step": step,
                    })
                    print("Step:{} Validation Kloss:{}".format(step, Kloss_val.item()))

            step += 1

    if len(val_losses) >= 10:
        convergence_loss = np.mean(val_losses[-10:])
    else:
        convergence_loss = np.mean(val_losses) if len(val_losses) > 0 else None

    print("END - Best loss: {}  Convergence loss: {}".format(best_loss, convergence_loss))
    wandb.log({"best_loss": best_loss, "convergence_loss": convergence_loss})
    wandb.finish()

def main():
    cov_regs = [0, 1]
    encode_dims = [1, 4, 16, 64, 256, 1024]
    random_seeds = [1, 2, 3]#[1, 2, 3]
    envs = ['Franka', 'LogisticMap']#['Polynomial', 'LogisticMap', 'DampingPendulum', 'DoublePendulum', 'Franka', 'G1', 'Go2']
    train_steps = {'G1': 20000, 'Go2': 20000, 'Franka': 80000, 'DoublePendulum': 60000, 
                   'DampingPendulum': 60000, 'Polynomial': 100000, 'LogisticMap': 100000}

    for random_seed, env, encode_dim, cov_reg in itertools.product(random_seeds, envs, encode_dims, cov_regs):
        if env == "Polynomial" or env == "LogisticMap":
            Ksteps = 1
        else:
            Ksteps = 15

        if env == "Franka" or env == "LogisticMap":
            normalize = False
        else:
            normalize = True

        if env == "Franka" or env == "LogisticMap":
            max_norm = 0.01
        else:
            max_norm = 1

        if env == "G1" or env == "Go2":
            layer_depth = 3
        elif env == "DoublePendulum":
            layer_depth = 5
        else:
            layer_depth = None
        
        train(project_name=f'Koopman_Results_Apr_3',
              env_name=env,
              train_samples=60000,
              val_samples=20000,
              test_samples=20000,
              Ksteps=Ksteps,
              train_steps=train_steps[env],
              encode_dim=encode_dim,
              layer_depth=layer_depth,
              cov_reg=cov_reg,
              gamma=0.8,
              seed=random_seed,
              batch_size=64,
              val_step=1000,
              initial_lr=1e-3,
              lr_step=100,
              lr_gamma=0.99,
              max_norm=max_norm,
              cov_reg_weight=1,
              normalize=normalize,
              )

if __name__ == "__main__":
    main()