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


def get_layers(input_dim, target_dim, num_hidden_layers=1, hidden_dim=256):
    layers = [input_dim]
    for _ in range(num_hidden_layers):
        layers.append(hidden_dim)
    layers.append(target_dim)
    return layers


def Klinear_loss(data, net, mse_loss, u_dim, gamma, device, all_loss=False):
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
    beta = 1.0
    beta_sum = 0.0
    loss = torch.zeros(1, dtype=torch.float64, device=device)

    for i in range(steps - 1):
        u_i = None if u_seq is None else u_seq[i]
        X_encoded = net.forward(X_encoded, u_i)

        beta_sum += beta
        if not all_loss:
            target = X_gt[i]
        else:
            target = net.encode(data[i + 1, :, u_dim:]) if u_dim is not None else net.encode(data[i + 1, :, :])

        loss += beta * mse_loss(X_encoded[:, :state_dim], target[:, :state_dim])
        beta *= gamma

    loss /= beta_sum
    return loss, initial_encoding


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

# ----------------------------
# Effective sample-size mapping (for display/analysis)
# ----------------------------
def effective_samples(env: str, sample_size: int) -> int:
    if env == 'G1':
        return int(sample_size * 200000 / 60000)
    elif env == 'Go2':
        return int(sample_size * 140000 / 60000)
    return sample_size


def train(project_name, env_name, train_samples=60000, val_samples=20000, test_samples=20000, Ksteps=15,
          train_steps=20000, encode_dim=16, hidden_layers=2, hidden_dim=256, gamma=0.99, seed=42, batch_size=64,
          initial_lr=1e-3, lr_step=1000, lr_gamma=0.95, val_step=1000, max_norm=1, normalize=False, use_residual=True,
          use_control_loss=True, use_covariance_loss=False, cov_loss_weight=1, ctrl_loss_weight=1, all_loss=False, m=100,
          multiply_encode_by_input_dim=True):
    """
    multiply_encode_by_input_dim=True -> actual latent size = encode_dim * state_dim
    multiply_encode_by_input_dim=False -> actual latent size = encode_dim (absolute)
    """

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

    data_collector = KoopmanDatasetCollector(env_name, train_samples, val_samples, test_samples, Ksteps, normalize=normalize, m=m)
    Ktrain_data, Kval_data, Ktest_data = map(lambda x: torch.from_numpy(x).float(), data_collector.get_data())

    u_dim = data_collector.u_dim
    state_dim = data_collector.state_dim

    actual_encode_dim = encode_dim * state_dim if multiply_encode_by_input_dim else encode_dim

    layers = get_layers(state_dim, actual_encode_dim, hidden_layers, hidden_dim)
    Nkoopman = state_dim + actual_encode_dim

    print('Environment:', env_name)
    print("u_dim:", u_dim)
    print("state_dim:", state_dim)
    print("Train data shape:", Ktrain_data.shape)
    print("Validation data shape:", Kval_data.shape)
    print("Test data shape:", Ktest_data.shape)
    print("Encoder layers:", layers)
    print(f"Encode dim param: {encode_dim} | Actual encode dim used: {actual_encode_dim} "
          f"({'per-state x input_dim' if multiply_encode_by_input_dim else 'absolute'})")

    net = KoopmanNet(layers, Nkoopman, u_dim, use_residual=use_residual).to(device)
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    mode_tag = "per_state" if multiply_encode_by_input_dim else "absolute"
    wandb.init(project=project_name,
               name=f"{env_name}_edim{actual_encode_dim}_{mode_tag}_covloss_{use_covariance_loss}_ctrlloss_{use_control_loss}_seed{seed}",
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

            Kloss, initial_encoding = Klinear_loss(X, net, mse_loss, u_dim, gamma, device, all_loss=all_loss)
            Ctrlloss = control_loss(X, net, mse_loss, u_dim, gamma, device) if (u_dim and use_control_loss) else torch.zeros(1, device=device)
            Closs = cov_loss(initial_encoding[:, state_dim:])

            loss = Kloss + (ctrl_loss_weight * Ctrlloss if use_control_loss else 0.0)

            if use_covariance_loss:
                d = initial_encoding[:, state_dim:].shape[1]
                denom = d * (d - 1)
                if denom > 0:
                    loss = loss + cov_loss_weight * Closs / denom

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm)
            optimizer.step()
            scheduler.step()

            wandb.log({"Train/Kloss": Kloss.item(), "Train/ControlLoss": Ctrlloss.item(), "step": step})

            if step % val_step == 0:
                with torch.no_grad():
                    Kloss_val, initial_encoding_val = Klinear_loss(Kval_data, net, mse_loss, u_dim, gamma, device, all_loss=all_loss)
                    Ctrlloss_val = control_loss(Kval_data, net, mse_loss, u_dim, gamma, device) if (u_dim and use_control_loss) else torch.zeros(1, device=device)
                    Closs_val = cov_loss(initial_encoding_val[:, state_dim:])
                    val_loss = Kloss_val

                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_state_dict = copy.deepcopy(net.state_dict())
                        best_model_path = os.path.join(model_dir, f"{timestamp}_model_{env_name}.pth")
                        norm_stats = getattr(data_collector, "norm_stats", None)
                        torch.save({
                            'model': best_state_dict,
                            'layer': layers,
                            'norm_stats': norm_stats,
                            'env_name': env_name,
                        }, best_model_path)

                    val_losses.append(val_loss.item())
                    wandb.log({
                        "Val/Kloss": Kloss_val.item(),
                        "Val/ControlLoss": Ctrlloss_val.item(),
                        "Val/CovLoss": Closs_val.item(),
                        "Val/best_Kloss": best_loss.item(),
                        "step": step
                    })
                    print(f"Step {step}: Val Kloss: {Kloss_val.item()}")

            step += 1

        convergence_loss = np.mean(val_losses[-10:]) if len(val_losses) >= 10 else np.mean(val_losses)
    wandb.log({"best_loss": best_loss.item(), "convergence_loss": convergence_loss})

    assert best_model_path is not None and os.path.exists(best_model_path), "Best model not saved properly."
    checkpoint = torch.load(best_model_path, weights_only=False, map_location=device)
    net.load_state_dict(checkpoint['model'])
    net.eval()

    with torch.no_grad():
        Ktest_loss, encoding_test = Klinear_loss(Ktest_data, net, mse_loss, u_dim, gamma, device, all_loss=all_loss)
        if u_dim is not None:
            Ctest_loss = control_loss(Ktest_data, net, mse_loss, u_dim, gamma, device)
        Ctest_cov = cov_loss(encoding_test[:, state_dim:])

    log_to_csv(csv_log_path, {
        "env_name": env_name,
        "state_dim": state_dim,
        "u_dim": u_dim,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
        "Ksteps": Ksteps,
        "train_steps": train_steps,
        "batch_size": batch_size,
        "initial_lr": initial_lr,
        "lr_step": lr_step,
        "lr_gamma": lr_gamma,
        "max_norm": max_norm,
        "val_step": val_step,
        "gamma": gamma,
        "use_residual": use_residual,
        "use_control_loss": use_control_loss,
        "use_covariance_loss": use_covariance_loss,
        "cov_loss_weight": cov_loss_weight,
        "all_loss": all_loss,
        "encode_dim_param": encode_dim,   # raw param passed in
        "encode_dim": actual_encode_dim,  # actual latent size used
        "encode_dim_mode": "times_input_dim" if multiply_encode_by_input_dim else "absolute",
        "hidden_layers": hidden_layers,
        "hidden_dim": hidden_dim,
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
        "m": m
    })

    wandb.finish()


def main():
    encode_dims = [1, 2, 4, 8, 16]
    sample_sizes = [4000, 16000]#[1000, 4000, 16000, 60000]
    layer_depths = [3]
    hidden_dims = [256]
    residuals = [True]
    control_losses = [False, True]
    ctrl_loss_weights = [1]
    covariance_losses = [False, True]
    cov_loss_weights = [1]
    random_seeds = [17382, 76849, 20965, 84902, 51194]
    envs = ['Franka']#["DampingPendulum", "DoublePendulum", "Franka", "Kinova", "G1", "Go2", "Polynomial"]
    train_steps = {'G1': 20000, 'Go2': 20000, 'Franka': 60000, 'DoublePendulum': 60000,
                   'DampingPendulum': 60000, 'Polynomial': 80000, 'Kinova': 60000}
    project_name = 'Aug_8'
    ms = [100]

    # Toggle/sweep this list to try both modes
    encode_dim_modes = [True]  # set to [True, False] to sweep both

    for random_seed, env, encode_dim, layer_depth, hidden_dim, residual, control_loss, covariance_loss, ctrl_loss_weight, cov_loss_weight, m, mult_by_input, sample_size in \
        itertools.product(random_seeds, envs, encode_dims, layer_depths, hidden_dims, residuals, control_losses, covariance_losses, ctrl_loss_weights, cov_loss_weights, ms, encode_dim_modes, sample_sizes):

        if env in ['Polynomial', 'LogisticMap']:
            Ksteps = 1
            if control_loss:
                continue
        else:
            Ksteps = 15

        if env in ['G1', 'Go2']:
            gamma = 0.99
            normalize = True
        else:
            gamma = 0.8
            normalize = False

        if env == 'G1':
            sample_size = int(sample_size * 200000/60000)
        elif env == 'Go2':
            sample_size = int(sample_size * 140000/60000)

        train(project_name=project_name,
              env_name=env,
              train_samples=sample_size,
              val_samples=20000,
              test_samples=20000,
              Ksteps=Ksteps,
              train_steps=train_steps[env],
              encode_dim=encode_dim,
              hidden_layers=layer_depth,
              hidden_dim=hidden_dim,
              gamma=gamma,
              seed=random_seed,
              batch_size=64,
              val_step=1000,
              initial_lr=1e-3,
              lr_step=1000,
              lr_gamma=0.9,
              max_norm=0.1,
              normalize=normalize,
              use_residual=residual,
              use_control_loss=control_loss,
              use_covariance_loss=covariance_loss,
              cov_loss_weight=cov_loss_weight,
              ctrl_loss_weight=ctrl_loss_weight,
              all_loss=False,
              m=m,
              multiply_encode_by_input_dim=mult_by_input
              )


if __name__ == "__main__":
    main()