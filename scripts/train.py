import torch
import numpy as np
import torch.nn as nn
import random
import copy
import itertools
import wandb
from torch.utils.data import DataLoader
import os
import sys
sys.path.append('../utility')
from dataset import KoopmanDatasetCollector, KoopmanDataset
from network import KoopmanNet

def get_layers(input_dim, target_dim, num_hidden_layers=1, hidden_dim=256):
    layers = [input_dim]
    for i in range(num_hidden_layers):
        layers.append(hidden_dim)
    layers.append(target_dim)
    return layers

def Klinear_loss(data, net, mse_loss, u_dim, gamma, device):
    steps, traj_num, N = data.shape
    state_dim = N if u_dim is None else N - u_dim

    if u_dim is None:
        u_seq = None
        X_gt = data[1:, :, :]
        X_encoded = net.encode(data[0])
    else:
        u_seq = data[:-1, :, :u_dim]
        X_gt = data[1:, :, u_dim:]
        X_encoded = net.encode(data[0, :, u_dim:])

    initial_encoding = X_encoded
    latent_seq = [X_encoded]

    for i in range(steps - 1):
        u_i = None if u_seq is None else u_seq[i]
        X_encoded = net.forward(X_encoded, u_i)
        latent_seq.append(X_encoded)

    Z_pred = torch.stack(latent_seq, dim=0)

    Z_pred = Z_pred[1:, :, :state_dim]

    squared_error = (Z_pred - X_gt) ** 2
    loss_per_step = squared_error.mean(dim=(1, 2))

    betas = gamma ** torch.arange(steps - 1, device=device).float()
    beta_sum = betas.sum()
    weighted_loss = (loss_per_step * betas).sum()

    return weighted_loss / beta_sum, initial_encoding

def control_loss(data, net, mse_loss, u_dim, gamma, device):
    steps, traj_num, N = data.shape
    state_dim = N - u_dim
    A = net.lA.weight
    B = net.lB.weight
    B_pinv_T = torch.linalg.pinv(B).t()

    betas = gamma ** torch.arange(steps - 1, device=device).float()
    beta_sum = betas.sum()

    encoded_states = net.encode(data[:, :, u_dim:])

    X_i     = encoded_states[:-1]
    X_ip1   = encoded_states[1:]
    u       = data[:-1, :, :u_dim]

    AX_i = torch.matmul(X_i, A.t())
    residual = X_ip1 - AX_i

    u_rec = torch.matmul(residual, B_pinv_T)

    mse_elementwise = (u_rec - u) ** 2
    loss_per_step = mse_elementwise.mean(dim=(1, 2))

    weighted_loss = (loss_per_step * betas).sum()
    return weighted_loss / beta_sum

def cov_loss(z):
    z_mean = torch.mean(z, dim=0, keepdim=True)
    z_centered = z - z_mean
    cov_matrix = (z_centered.t() @ z_centered) / (z_centered.size(0) - 1)
    diag_cov = torch.diag(torch.diag(cov_matrix))
    off_diag = cov_matrix - diag_cov
    return torch.norm(off_diag, p='fro')**2

def train(project_name, env_name, train_samples=60000, val_samples=20000, test_samples=20000, Ksteps=15,
          train_steps=20000, encode_dim=16, hidden_layers=2, gamma=0.99, seed=42, batch_size=64, 
          initial_lr=1e-3, lr_step=1000, lr_gamma=0.95, val_step=1000, max_norm=1, normalize=False):

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

    data_collector = KoopmanDatasetCollector(env_name, train_samples, val_samples, test_samples, Ksteps, normalize=normalize)
    Ktrain_data, Kval_data, Ktest_data = data_collector.get_data()

    Ktrain_data = torch.from_numpy(Ktrain_data).float()
    Kval_data = torch.from_numpy(Kval_data).float()

    u_dim = data_collector.u_dim
    state_dim = data_collector.state_dim

    print("u_dim:", u_dim)
    print("state_dim:", state_dim)

    print("Train data shape:", Ktrain_data.shape)
    print("Validation data shape:", Kval_data.shape)

    layers = get_layers(state_dim, encode_dim, hidden_layers)
    Nkoopman = state_dim + encode_dim

    print("Encoder layers:", layers)

    net = KoopmanNet(layers, Nkoopman, u_dim)
    net.to(device)
    mse_loss = nn.MSELoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=initial_lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    wandb.init(project=project_name, 
               name=f"{env_name}_edim{encode_dim}_seed{seed}",
               config={
                    "env_name": env_name,
                    "train_steps": train_steps,
                    "encode_dim": encode_dim,
                    "hidden_layers": hidden_layers,
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

            if u_dim is not None:
                Ctrlloss = control_loss(X, net, mse_loss, u_dim, gamma, device)
            else:
                Ctrlloss = torch.zeros(1, dtype=torch.float32).to(device)

            loss = Kloss + Ctrlloss

            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(net.parameters(), max_norm)

            optimizer.step()
            scheduler.step()

            wandb.log({
                "Train/Kloss": Kloss.item(),
                "Train/ControlLoss": Ctrlloss.item() if u_dim is not None else 0,
                "step": step
            })

            if step % val_step == 0:
                with torch.no_grad():
                    Kloss_val, initial_encoding = Klinear_loss(Kval_data, net, mse_loss, u_dim, gamma, device)
                    Closs_val = cov_loss(initial_encoding[:, state_dim:])
                    if u_dim is not None:
                        Ctrlloss_val = control_loss(Kval_data, net, mse_loss, u_dim, gamma, device)
                    else:
                        Ctrlloss_val = torch.zeros(1, dtype=torch.float32).to(device)
                    
                    loss_val = Kloss_val #+ Ctrlloss_val

                    val_losses.append(loss_val.item())

                    if loss_val < best_loss:
                        best_loss = copy.copy(loss_val)
                        best_state_dict = copy.copy(net.state_dict())
                        saved_dict = {'model':best_state_dict,'layer':layers}
                        torch.save(saved_dict, f"../log/best_models/{project_name}/best_model_{norm_str}_{env_name}_{encode_dim}_{seed}.pth")

                    wandb.log({
                        "Val/Kloss": Kloss_val.item(),
                        "Val/CovLoss": Closs_val.item(),
                        "Val/ControlLoss": Ctrlloss_val.item(),
                        "Val/best_Kloss": best_loss.item() if torch.is_tensor(best_loss) else best_loss,
                        "step": step,
                    })
                    print("Step:{} Validation Kloss:{}".format(step, Kloss_val.item()))

            step += 1

    if len(val_losses) >= 10:
        convergence_loss = np.mean(val_losses[-10:])
    else:
        convergence_loss = np.mean(val_losses) if len(val_losses) > 0 else None

    print("END - Best loss: {}  Convergence loss: {}".format(best_loss.item(), convergence_loss))
    wandb.log({"best_loss": best_loss.item(), "convergence_loss": convergence_loss})
    wandb.finish()

def main():
    encode_dims = [4, 16, 64, 256, 1024]
    random_seeds = [1]
    envs = ['G1', 'Go2']#['Polynomial', 'LogisticMap', 'DampingPendulum', 'DoublePendulum', 'Kinova', 'G1', 'Go2']
    train_steps = {'G1': 20000, 'Go2': 20000, 'Kinova': 60000, 'Franka': 60000, 'DoublePendulum': 60000, 
                   'DampingPendulum': 60000, 'Polynomial': 80000, 'LogisticMap': 80000, 'CartPole': 60000,
                   'MountainCarContinuous': 60000}
    project_name = 'Koopman_Results_Apr_29'

    for random_seed, env, encode_dim in itertools.product(random_seeds, envs, encode_dims):
        train(project_name=project_name,
              env_name=env,
              train_samples=60000,
              val_samples=20000,
              test_samples=20000,
              Ksteps=15,
              train_steps=train_steps[env],
              encode_dim=encode_dim,
              hidden_layers=3,
              gamma=0.99,
              seed=random_seed,
              batch_size=64,
              val_step=1000,
              initial_lr=1e-3,
              lr_step=1000,
              lr_gamma=0.9,
              max_norm=0.1,
              normalize=True,
              )

if __name__ == "__main__":
    main()