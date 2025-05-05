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
import csv
from datetime import datetime
sys.path.append('../utility')
from dataset import KoopmanDatasetCollector, KoopmanDataset
from network import KoopmanNet

def get_doubling_layers(input_dim, target_dim):
    layers = [input_dim]
    while layers[-1] < target_dim:
        next_width = layers[-1] * 2
        if next_width > target_dim:
            next_width = target_dim
        layers.append(next_width)
    if len(layers) == 1:
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def log_to_csv(log_path, log_dict):
    file_exists = os.path.isfile(log_path)
    with open(log_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_dict)

def train(project_name, env_name, train_samples=60000, val_samples=20000, test_samples=20000, Ksteps=15,
          train_steps=20000, encode_dim=16, gamma=0.99, seed=42, batch_size=64, initial_lr=1e-3, lr_step=1000, 
          lr_gamma=0.95, val_step=1000, max_norm=1, normalize=False, use_residual=True, cov_reg=True, cov_reg_weight=1):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = f"../log/{project_name}/best_models/"
    os.makedirs(model_dir, exist_ok=True)
    csv_log_path = f"../log/{project_name}/koopman_results_log.csv"

    data_collector = KoopmanDatasetCollector(env_name, train_samples, val_samples, test_samples, Ksteps, normalize=normalize)
    Ktrain_data, Kval_data, Ktest_data = map(lambda x: torch.from_numpy(x).float(), data_collector.get_data())

    u_dim = data_collector.u_dim
    state_dim = data_collector.state_dim
    layers = get_doubling_layers(state_dim, encode_dim)

    Nkoopman = state_dim + encode_dim

    print('Environment:', env_name)
    print("u_dim:", u_dim)
    print("state_dim:", state_dim)
    print("Train data shape:", Ktrain_data.shape)
    print("Validation data shape:", Kval_data.shape)
    print("Test data shape:", Ktest_data.shape)
    print("Encoder layers:", layers)

    net = KoopmanNet(layers, Nkoopman, u_dim, use_residual=use_residual).to(device)
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    wandb.init(project=project_name,
               name=f"{env_name}_edim{encode_dim}_covreg_{cov_reg}_seed{seed}",
               config=locals())

    best_loss = 1e10
    val_losses = []
    train_loader = DataLoader(KoopmanDataset(Ktrain_data), batch_size=batch_size, shuffle=True, pin_memory=True)
    Kval_data = Kval_data.to(device)
    Ktest_data = Ktest_data.to(device)
    step = 0

    best_model_path = None

    while step < train_steps:
        for batch in train_loader:
            if step >= train_steps:
                break
            X = batch.permute(1, 0, 2).to(device)
            Kloss, initial_encoding = Klinear_loss(X, net, mse_loss, u_dim, gamma, device)
            Ctrlloss = control_loss(X, net, mse_loss, u_dim, gamma, device) if u_dim else torch.zeros(1, device=device)
            Covloss = cov_loss(initial_encoding[:, state_dim:]) if cov_reg else torch.zeros(1, device=device)
            loss = Kloss + Ctrlloss
            if cov_reg:
                loss += cov_reg_weight * Covloss / (initial_encoding[:, state_dim:].shape[1] * (initial_encoding[:, state_dim:].shape[1] - 1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm)
            optimizer.step()
            scheduler.step()

            wandb.log({"Train/Kloss": Kloss.item(), 
                        "Train/ControlLoss": Ctrlloss.item(), 
                        "Train/CovLoss": Covloss.item(),
                        "step": step})

            if step % val_step == 0:
                with torch.no_grad():
                    Kloss_val, initial_encoding = Klinear_loss(Kval_data, net, mse_loss, u_dim, gamma, device)
                    Ctrlloss_val = control_loss(Kval_data, net, mse_loss, u_dim, gamma, device) if u_dim else torch.zeros(1, device=device)
                    Covloss_val = cov_loss(initial_encoding[:, state_dim:])
                    val_loss = Kloss_val

                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_state_dict = copy.deepcopy(net.state_dict())
                        best_model_path = os.path.join(model_dir, f"{timestamp}_model_{env_name}.pth")
                        torch.save({'model': best_state_dict, 'layer': layers}, best_model_path)

                    val_losses.append(val_loss.item())
                    wandb.log({
                        "Val/Kloss": Kloss_val.item(),
                        "Val/ControlLoss": Ctrlloss_val.item(),
                        "Val/CovLoss": Covloss_val.item(),
                        "Val/best_Kloss": best_loss.item(),
                        "step": step
                    })
                    print(f"Step {step}: Val Kloss: {Kloss_val.item()}")

            step += 1

        convergence_loss = np.mean(val_losses[-10:]) if len(val_losses) >= 10 else np.mean(val_losses)
    wandb.log({"best_loss": best_loss.item(), "convergence_loss": convergence_loss})

    assert best_model_path is not None and os.path.exists(best_model_path), "Best model not saved properly."
    checkpoint = torch.load(best_model_path, map_location=device)
    net.load_state_dict(checkpoint['model'])
    net.eval()

    with torch.no_grad():
        Ktest_loss, encoding_test = Klinear_loss(Ktest_data, net, mse_loss, u_dim, gamma, device)
        if u_dim is not None:
            Ctest_loss = control_loss(Ktest_data, net, mse_loss, u_dim, gamma, device)
        Ctest_cov = cov_loss(encoding_test[:, state_dim:])

    log_to_csv(csv_log_path, {
        "env_name": env_name,
        "encode_dim": encode_dim,
        "cov_reg": cov_reg,
        "cov_reg_weight": cov_reg_weight,
        "seed": seed,
        "normalize": normalize,
        "best_val_Kloss": best_loss.item(),
        "convergence_val_Kloss": convergence_loss,
        "test_Kloss": Ktest_loss.item(),
        "test_CovLoss": Ctest_cov.item(),
        "test_ControlLoss": Ctest_loss.item() if u_dim is not None else np.nan,
        "model_path": best_model_path,
        "num_params": count_parameters(net),
        "encoder_num_params": count_parameters(net.encode_net),
    })

    wandb.finish()

def main():
    encode_dims = [4, 16, 64, 256, 1024]
    cov_reg_weights = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    cov_regs = [True]
    random_seeds = [2,3,4,5]
    envs = ['Polynomial', 'LogisticMap', 'DampingPendulum', 'DoublePendulum', 'Franka', 'G1', 'Go2']
    train_steps = {'G1': 20000, 'Go2': 20000, 'Kinova': 60000, 'Franka': 60000, 'DoublePendulum': 60000, 
                   'DampingPendulum': 60000, 'Polynomial': 80000, 'LogisticMap': 80000, 'CartPole': 60000,
                   'MountainCarContinuous': 60000}
    project_name = 'May_2_Unfolding_CovReg'

    for random_seed, env, encode_dim, cov_reg, cov_reg_weight in itertools.product(random_seeds, envs, encode_dims, cov_regs, cov_reg_weights):
        train(project_name=project_name,
              env_name=env,
              train_samples=60000,
              val_samples=20000,
              test_samples=20000,
              Ksteps=15,
              train_steps=train_steps[env],
              encode_dim=encode_dim,
              cov_reg=cov_reg,
              cov_reg_weight=cov_reg_weight,
              gamma=0.99,
              seed=random_seed,
              batch_size=64,
              val_step=1000,
              initial_lr=1e-3,
              lr_step=1000,
              lr_gamma=0.9,
              max_norm=0.1,
              normalize=True,
              use_residual=True
              )

if __name__ == "__main__":
    main()