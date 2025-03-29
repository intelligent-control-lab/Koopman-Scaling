import torch
import numpy as np
import torch.nn as nn
import random
import copy
import itertools
import wandb
from torch.utils.data import DataLoader
import math
from dataset import KoopmanDatasetCollector, KoopmanDataset
from network import KoopmanNet, LearnedCovWeight

def get_layers(input_dim, target_dim, layer_depth=5):
    if layer_depth < 2:
        raise ValueError("Layer depth must be at least 2 (input and target layers).")

    layers = [input_dim]
    
    if input_dim < target_dim:
        factor = (target_dim / input_dim) ** (1 / (layer_depth - 1))
        for i in range(1, layer_depth - 1):
            layers.append(int(math.ceil(layers[-1] * factor)))
    else:
        factor = (target_dim / input_dim) ** (1 / (layer_depth - 1))
        for i in range(1, layer_depth - 1):
            layers.append(int(math.ceil(layers[-1] * factor)))

    layers.append(target_dim)
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
          train_steps=20000, all_loss=0, encode_dim=16, layer_depth=5, cov_reg=0, gamma=0.99, seed=42, 
          batch_size=64, initial_lr=1e-3, lr_step=1000, lr_gamma=0.95, val_step=1000, max_norm=1, cov_reg_weight_init=1e-3):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


    print("Loading dataset...")
    data_collector = KoopmanDatasetCollector(env_name, train_samples, val_samples, test_samples, Ksteps, device)
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

    layers = get_layers(state_dim, encode_dim, layer_depth)
    Nkoopman = state_dim + layers[-1]

    print("Encoder layers:", layers)

    net = KoopmanNet(layers, Nkoopman, u_dim)
    net.to(device)
    net.double()
    mse_loss = nn.MSELoss()

    if cov_reg:
        learned_cov_weight = LearnedCovWeight(init_val=cov_reg_weight_init).to(device).double()
        optimizer = torch.optim.Adam(
            list(net.parameters()) + list(learned_cov_weight.parameters()),
            lr=initial_lr
        )
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=initial_lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    
    for name, param in net.named_parameters():
        print("model:", name, param.requires_grad)

    wandb.init(project=project_name, 
               name=f"{env_name}_edim{encode_dim}_closs{'on' if cov_reg else 'off'}_seed{seed}",
               config={
                    "env_name": env_name,
                    "train_steps": train_steps,
                    "all_loss": all_loss,
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
                    "cov_reg_weight_init": cov_reg_weight_init,
               })

    best_loss = 1e10
    step = 0
    val_losses = []

    train_dataset = KoopmanDataset(Ktrain_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    Kval_data = Kval_data.to(device)

    while step < train_steps:
        for batch in train_loader:
            if step >= train_steps:
                break
            X = batch.permute(1, 0, 2).to(device)

            Kloss, initial_encoding = Klinear_loss(X, net, mse_loss, u_dim, gamma, device)

            Closs = cov_loss(initial_encoding)

            if cov_reg:
                factor = learned_cov_weight()
                loss = Kloss + factor * Closs
            else:
                loss = Kloss


            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(net.lA.parameters(), max_norm)
            if u_dim is not None:
                torch.nn.utils.clip_grad_norm_(net.lB.parameters(), max_norm)

            optimizer.step()
            scheduler.step()

            wandb.log({
                "Train/Kloss": Kloss.item(),
                "Train/CovLoss": Closs.item(),
                "Train/learned_cov_weight": factor.item() if cov_reg else 0,
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
                        torch.save(saved_dict, f"../log/best_models/best_model_{env_name}_{encode_dim}_{cov_reg}_{seed}.pth")

                    wandb.log({
                        "Val/Kloss": Kloss_val.item(),
                        "Val/CovLoss": Closs_val.item(),
                        "Val/best_Kloss": best_loss.item(),
                        "Val/learned_cov_weight": factor.item() if cov_reg else 0,
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
    encode_dims = [1, 4, 16, 64, 256, 512, 1024]
    random_seeds = [1]
    envs = ['G1', 'Go2', 'LogisticMap', 'DampingPendulum', 'DoublePendulum', 'Franka', 'Polynomial']
    #envs = ['LogisticMap']

    for env, encode_dim, cov_reg, random_seed in itertools.product(envs, encode_dims, cov_regs, random_seeds):
        if env == "Polynomial" or env == "LogisticMap":
            Ksteps = 1
        else:
            Ksteps = 15

        if env in ["Polynomial", "Franka", "DoublePendulum", "DampingPendulum"]:
            max_norm = 0.01
            cov_reg_weight_init = 1e-3
        elif env in ["LogisticMap"]:
            max_norm = 0.001
            cov_reg_weight_init = 1e-6
        elif env in ["G1"]:
            max_norm = 1
            cov_reg_weight_init = 1e-7
        elif env in ["Go2"]:
            max_norm = 1
            cov_reg_weight_init = 1e-6

        train(project_name=f'Koopman_{env}',
              env_name=env,
              train_samples=60000,
              val_samples=20000,
              test_samples=20000,
              Ksteps=Ksteps,
              train_steps=100000,
              encode_dim=encode_dim,
              layer_depth=5,
              cov_reg=cov_reg,
              gamma=0.8,
              seed=random_seed,
              batch_size=64,
              val_step=1000,
              initial_lr=1e-3,
              lr_step=100,
              lr_gamma=0.99,
              max_norm=max_norm,
              cov_reg_weight_init=cov_reg_weight_init)

if __name__ == "__main__":
    main()