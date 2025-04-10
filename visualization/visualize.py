import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../scripts')
sys.path.append('../utility')

from dataset import KoopmanDatasetCollector
from network import KoopmanNet

# ---------------- Utility Functions ----------------

def compute_cov_loss(z):
    """Compute the off-diagonal Frobenius norm of the covariance of z."""
    z_mean = torch.mean(z, dim=0, keepdim=True)
    z_centered = z - z_mean
    cov_matrix = (z_centered.t() @ z_centered) / (z_centered.size(0) - 1)
    diag_cov = torch.diag(torch.diag(cov_matrix))
    off_diag = cov_matrix - diag_cov
    return torch.norm(off_diag, p='fro')**2


def compute_overall_metrics(model, test_data, u_dim, gamma, device):
    """Compute weighted multi‑step prediction error & covariance loss."""
    mse_loss = torch.nn.MSELoss()
    test_data = test_data.to(device)

    if u_dim is None:
        steps, traj_num, state_dim = test_data.shape
        X_current = model.encode(test_data[0, :])
        initial_encoding = X_current
        beta = 1.0
        beta_sum = 0.0
        total_loss = 0.0
        for i in range(steps - 1):
            X_current = model.forward(X_current, None)
            beta_sum += beta
            total_loss += beta * mse_loss(X_current[:, :state_dim], test_data[i + 1, :])
            beta *= gamma
        overall_error = total_loss / beta_sum
    else:
        steps, traj_num, N = test_data.shape
        state_dim = N - u_dim
        X_current = model.encode(test_data[0, :, u_dim:])
        initial_encoding = X_current
        beta = 1.0
        beta_sum = 0.0
        total_loss = 0.0
        for i in range(steps - 1):
            X_current = model.forward(X_current, test_data[i, :, :u_dim])
            beta_sum += beta
            total_loss += beta * mse_loss(X_current[:, :state_dim], test_data[i + 1, :, u_dim:])
            beta *= gamma
        overall_error = total_loss / beta_sum

    cov_loss_value = compute_cov_loss(initial_encoding)
    return overall_error.item(), cov_loss_value.item(), initial_encoding.shape[1]


def compute_per_step_errors(model, test_data, u_dim, gamma, device):
    """Compute prediction error at each step."""
    mse_loss = torch.nn.MSELoss()
    test_data = test_data.to(device)

    if u_dim is None:
        steps, traj_num, state_dim = test_data.shape
        X_current = model.encode(test_data[0, :])
        errors = []
        for i in range(steps - 1):
            X_current = model.forward(X_current, None)
            error = mse_loss(X_current[:, :state_dim], test_data[i + 1, :])
            errors.append(error.item())
    else:
        steps, traj_num, N = test_data.shape
        state_dim = N - u_dim
        X_current = model.encode(test_data[0, :, u_dim:])
        errors = []
        for i in range(steps - 1):
            X_current = model.forward(X_current, test_data[i, :, :u_dim])
            error = mse_loss(X_current[:, :state_dim], test_data[i + 1, :, u_dim:])
            errors.append(error.item())
    return errors

# ---------------- Helper: log‑symmetric error band ----------------

def log_symmetric_band(y: np.ndarray, y_std: np.ndarray):
    """Return lower & upper envelopes symmetric in log space."""
    mask = np.isnan(y) | np.isnan(y_std) | (y <= 0)
    y_m = np.ma.masked_array(y, mask=mask)
    y_std_m = np.ma.masked_array(y_std, mask=mask)

    log_y = np.log10(y_m)
    log_y_std = np.log10(y_m + y_std_m) - log_y  # deviation in log10 space

    lower = 10 ** (log_y - log_y_std)
    upper = 10 ** (log_y + log_y_std)
    return lower, upper, y_m

# ---------------- Main Visualization Function ----------------

def main():
    project_name = "Koopman_Results_Apr_8_2"
    model_dir = f"../log/best_models/{project_name}/"
    fig_dir = f"{project_name}_figure"
    os.makedirs(fig_dir, exist_ok=True)

    envs = ['Polynomial']#['Polynomial', 'LogisticMap', 'DampingPendulum', 'DoublePendulum', 'Franka', 'G1', 'Go2']  # extend as needed
    random_seeds = [1]
    encode_dims = [1, 4, 16, 64, 256, 1024]
    cov_regs = [0, 1]
    gamma = 0.8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- storage ---
    results_overall = {e: {c: {d: [] for d in encode_dims} for c in cov_regs} for e in envs}
    results_cov = {e: {c: {d: [] for d in encode_dims} for c in cov_regs} for e in envs}
    results_per_step = {e: {c: {d: [] for d in encode_dims} for c in cov_regs} for e in envs}
    encode_dims_out = {e: {c: {} for c in cov_regs} for e in envs}

    # --- evaluation loop ---
    for env in envs:
        print(f"Processing {env} …")
        train_samples, val_samples, test_samples = 60000, 20000, 20000
        Ksteps = 1 if env in ["Polynomial", "LogisticMap"] else 10

        norm_str = "norm"#"nonorm" if env in ["Franka", "LogisticMap"] else "norm"
        normalize = norm_str == "norm"
        collector = KoopmanDatasetCollector(env, train_samples, val_samples, test_samples, Ksteps, normalize=normalize)
        test_data = torch.from_numpy(collector.test_data).float()
        #test_data = test_data[:12, :, :]  # only first 12 steps
        state_dim = collector.state_dim
        u_dim = collector.u_dim

        for seed in random_seeds:
            for cov_reg in cov_regs:
                for edim in encode_dims:
                    if edim == 1 and cov_reg == 1:
                        fname = os.path.join(model_dir, f"best_model_{norm_str}_{env}_{edim}_0_{seed}.pth")
                    else:
                        fname = os.path.join(model_dir, f"best_model_{norm_str}_{env}_{edim}_{cov_reg}_{seed}.pth")
                    if not os.path.exists(fname):
                        print("  ✗", fname)
                        continue
                    saved = torch.load(fname, map_location=device)
                    layers = saved['layer']
                    Nkoopman = state_dim + layers[-1]
                    model = KoopmanNet(layers, Nkoopman, u_dim)
                    model.load_state_dict(saved['model'])
                    model.to(device).eval()

                    overall, cov_loss, enc_out_dim = compute_overall_metrics(model, test_data, u_dim, gamma, device)
                    per_step = compute_per_step_errors(model, test_data, u_dim, gamma, device)

                    results_overall[env][cov_reg][edim].append(overall)
                    results_cov[env][cov_reg][edim].append(cov_loss)
                    results_per_step[env][cov_reg][edim].append(per_step)
                    encode_dims_out[env][cov_reg][edim] = enc_out_dim

    # --- plotting ---
    cmap = plt.get_cmap("viridis")
    color_indices = np.linspace(0, 1, len(encode_dims))

    for env in envs:
        # ---------- Overall prediction error ----------
        overall_means = {c: [np.nanmean(results_overall[env][c][d]) if results_overall[env][c][d] else np.nan for d in encode_dims] for c in cov_regs}
        overall_stds = {c: [np.nanstd(results_overall[env][c][d]) if results_overall[env][c][d] else np.nan for d in encode_dims] for c in cov_regs}
        x_vals = np.array(encode_dims, dtype=float)

        plt.figure(figsize=(8, 6))
        plt.xscale('log'); plt.yscale('log')
        for c in cov_regs:
            label = "Covariance Loss Off" if c == 0 else "Covariance Loss On"
            y = np.array(overall_means[c], dtype=float)
            y_std = np.array(overall_stds[c], dtype=float)
            lower, upper, y_masked = log_symmetric_band(y, y_std)
            plt.plot(x_vals, y_masked, marker='o', label=label)
            plt.fill_between(x_vals, lower, upper, alpha=0.3)
        plt.xlabel("Encode Dimension")
        plt.ylabel("Overall Multi‑step Prediction Error")
        plt.title(f"{env} — Overall Prediction Error")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"{env}_overall_error.png")); plt.close()

        # ---------- Normalized covariance loss ----------
        norm_cov_means, norm_cov_stds = {c: [] for c in cov_regs}, {c: [] for c in cov_regs}
        for c in cov_regs:
            for d in encode_dims:
                cov_vals = results_cov[env][c][d]
                norm_vals = []
                for cl in cov_vals:
                    enc_dim = encode_dims_out[env][c].get(d, None)
                    norm_vals.append(cl if enc_dim in (None, 0, 1) else cl / (enc_dim * (enc_dim - 1)))
                norm_cov_means[c].append(np.nanmean(norm_vals) if norm_vals else np.nan)
                norm_cov_stds[c].append(np.nanstd(norm_vals) if norm_vals else np.nan)

        plt.figure(figsize=(8, 6))
        plt.xscale('log'); plt.yscale('log')
        for c in cov_regs:
            label = "Covariance Loss Off" if c == 0 else "Covariance Loss On"
            y = np.array(norm_cov_means[c], dtype=float)
            y_std = np.array(norm_cov_stds[c], dtype=float)
            lower, upper, y_masked = log_symmetric_band(y, y_std)
            plt.plot(x_vals, y_masked, marker='o', label=label)
            plt.fill_between(x_vals, lower, upper, alpha=0.3)
        plt.xlabel("Encode Dimension")
        plt.ylabel("Normalized Covariance Loss")
        plt.title(f"{env} — Normalized Covariance Loss")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"{env}_normalized_cov_loss.png")); plt.close()

        # ---------- Per‑step error curves (log log‑symmetric bands) ----------
        plt.figure(figsize=(10, 6))
        plt.yscale('log')
        for i, d in enumerate(encode_dims):
            color = cmap(color_indices[i])
            for c in cov_regs:
                curves = results_per_step[env][c][d]
                if not curves:
                    continue
                arr = np.array(curves)
                mean_curve = np.mean(arr, axis=0)
                std_curve = np.std(arr, axis=0)
                steps = np.arange(1, len(mean_curve) + 1)
                linestyle = '-' if c == 0 else '--'
                label = f"edim={d}, {'No CL' if c == 0 else 'CL'}"
                lower, upper, mean_curve_masked = log_symmetric_band(mean_curve, std_curve)
                plt.plot(steps, mean_curve_masked, color=color, linestyle=linestyle, label=label)
                plt.fill_between(steps, lower, upper, color=color, alpha=0.3)
        plt.xlabel("Step"); plt.ylabel("Prediction Error")
        plt.title(f"{env} — Per‑step Prediction Error")
        plt.legend(fontsize='small', ncol=2); plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"{env}_per_step_error.png")); plt.close()


if __name__ == "__main__":
    main()
