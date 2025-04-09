# import os
# import glob
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import sys
# sys.path.append('../scripts')

# from dataset import KoopmanDatasetCollector
# from network import KoopmanNet

# # ---------------- Utility Functions ----------------

# def compute_cov_loss(z):
#     """Compute the off-diagonal Frobenius norm of the covariance of z."""
#     z_mean = torch.mean(z, dim=0, keepdim=True)
#     z_centered = z - z_mean
#     cov_matrix = (z_centered.t() @ z_centered) / (z_centered.size(0) - 1)
#     diag_cov = torch.diag(torch.diag(cov_matrix))
#     off_diag = cov_matrix - diag_cov
#     return torch.norm(off_diag, p='fro')**2

# def compute_overall_metrics(model, test_data, u_dim, gamma, device):
#     """
#     Computes the overall weighted multi-step prediction error and covariance loss.
#     Returns:
#       - overall_error: weighted prediction error (scalar)
#       - cov_loss_value: covariance loss on the initial encoding (scalar)
#       - encode_out_dim: dimension of the encoded features (used for normalization)
#     """
#     mse_loss = torch.nn.MSELoss()
#     test_data = test_data.to(device)
    
#     if u_dim is None:
#         # For environments with only state data.
#         steps, traj_num, state_dim = test_data.shape
#         X_current = model.encode(test_data[0, :])
#         initial_encoding = X_current
#         beta = 1.0
#         beta_sum = 0.0
#         total_loss = 0.0
#         for i in range(steps - 1):
#             X_current = model.forward(X_current, None)
#             beta_sum += beta
#             total_loss += beta * mse_loss(X_current[:, :state_dim], test_data[i + 1, :])
#             beta *= gamma
#         overall_error = total_loss / beta_sum
#     else:
#         # For environments with both action and state data.
#         steps, traj_num, N = test_data.shape
#         state_dim = N - u_dim
#         X_current = model.encode(test_data[0, :, u_dim:])
#         initial_encoding = X_current
#         beta = 1.0
#         beta_sum = 0.0
#         total_loss = 0.0
#         for i in range(steps - 1):
#             X_current = model.forward(X_current, test_data[i, :, :u_dim])
#             beta_sum += beta
#             total_loss += beta * mse_loss(X_current[:, :state_dim], test_data[i + 1, :, u_dim:])
#             beta *= gamma
#         overall_error = total_loss / beta_sum

#     cov_loss_value = compute_cov_loss(initial_encoding)
#     return overall_error.item(), cov_loss_value.item(), initial_encoding.shape[1]

# def compute_per_step_errors(model, test_data, u_dim, gamma, device):
#     """
#     Computes the prediction error at each prediction step.
#     Returns a list of errors (one per step).
#     """
#     mse_loss = torch.nn.MSELoss()
#     test_data = test_data.to(device)
    
#     if u_dim is None:
#         steps, traj_num, state_dim = test_data.shape
#         X_current = model.encode(test_data[0, :])
#         errors = []
#         for i in range(steps - 1):
#             X_current = model.forward(X_current, None)
#             error = mse_loss(X_current[:, :state_dim], test_data[i + 1, :])
#             errors.append(error.item())
#     else:
#         steps, traj_num, N = test_data.shape
#         state_dim = N - u_dim
#         X_current = model.encode(test_data[0, :, u_dim:])
#         errors = []
#         for i in range(steps - 1):
#             X_current = model.forward(X_current, test_data[i, :, :u_dim])
#             error = mse_loss(X_current[:, :state_dim], test_data[i + 1, :, u_dim:])
#             errors.append(error.item())
#     return errors

# # ---------------- Main Visualization Function ----------------

# def main():
#     # Directories and parameters
#     project_name = "Koopman_Results_Apr_3"
#     model_dir = f"../log/best_models/{project_name}/"
#     fig_dir = "figure"
#     os.makedirs(fig_dir, exist_ok=True)
    
#     # Environments and parameters
#     envs = ['LogisticMap']#['Polynomial', 'LogisticMap', 'DampingPendulum', 'DoublePendulum', 'Franka', 'G1', 'Go2']
#     random_seeds = [2,3,4,7]
#     encode_dims = [1, 4, 16, 64, 256, 1024]
#     cov_regs = [0, 1]  # 0: covariance loss disabled, 1: enabled
#     gamma = 0.8  # as used in training
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Data structures to accumulate metrics.
#     # For each environment, cov_reg option, and encode_dim, we store:
#     #   - overall prediction error (scalar)
#     #   - covariance loss (scalar)
#     #   - per-step prediction error (list of errors over steps)
#     results_overall = {env: {cov: {edim: [] for edim in encode_dims} for cov in cov_regs} for env in envs}
#     results_cov = {env: {cov: {edim: [] for edim in encode_dims} for cov in cov_regs} for env in envs}
#     results_per_step = {env: {cov: {edim: [] for edim in encode_dims} for cov in cov_regs} for env in envs}
#     # Also store the encoding output dimension (for normalization) per model.
#     encode_dims_out = {env: {cov: {} for cov in cov_regs} for env in envs}
    
#     # Loop over environments
#     for env in envs:
#         print(f"Processing environment: {env}")
#         # For test data, we use the same numbers as in training.
#         train_samples, val_samples, test_samples = 60000, 20000, 20000
#         # Set Ksteps: if env is "Polynomial" or "LogisticMap", use Ksteps=1; else 15.
#         Ksteps = 1 if env in ["Polynomial", "LogisticMap"] else 15
        
#         # Instantiate dataset collector and get test data.
#         norm_str = "nonorm" if env in ["Franka", "LogisticMap"] else "norm"
#         normalize = True if norm_str == "norm" else False
#         collector = KoopmanDatasetCollector(env, train_samples, val_samples, test_samples, Ksteps, device, normalize=normalize)
#         test_data = torch.from_numpy(collector.test_data).double()
#         state_dim = collector.state_dim
#         u_dim = collector.u_dim  # may be None
        
#         # Loop over seeds, cov_reg options and encode_dims
#         for seed in random_seeds:
#             for cov_reg in cov_regs:
#                 for edim in encode_dims:
#                     # Construct the file name (assuming normalization was used)
#                     model_filename = os.path.join(model_dir, f"best_model_{norm_str}_{env}_{edim}_{cov_reg}_{seed}.pth")
#                     if not os.path.exists(model_filename):
#                         print(f"File not found: {model_filename}. Skipping.")
#                         continue
#                     # Load the saved model
#                     saved = torch.load(model_filename, map_location=device)
#                     layers = saved['layer']
#                     Nkoopman = state_dim + layers[-1]
#                     model = KoopmanNet(layers, Nkoopman, u_dim)
#                     model.load_state_dict(saved['model'])
#                     model.to(device)
#                     model.double()
#                     model.eval()
                    
#                     # Evaluate overall metrics and per-step errors on the test set.
#                     overall_err, cov_loss_val, enc_out_dim = compute_overall_metrics(model, test_data, u_dim, gamma, device)
#                     per_step_err = compute_per_step_errors(model, test_data, u_dim, gamma, device)
                    
#                     results_overall[env][cov_reg][edim].append(overall_err)
#                     results_cov[env][cov_reg][edim].append(cov_loss_val)
#                     results_per_step[env][cov_reg][edim].append(per_step_err)
#                     encode_dims_out[env][cov_reg][edim] = enc_out_dim

#     # ---------------- Plotting ----------------
#     # Set up a colormap for different encode dimensions (for per-step plots)
#     cmap = plt.get_cmap("viridis")
#     color_indices = np.linspace(0, 1, len(encode_dims))
    
#     for env in envs:
#         # Prepare data for Set A: Overall prediction error vs encode dimension.
#         overall_means = {cov: [] for cov in cov_regs}
#         overall_stds = {cov: [] for cov in cov_regs}
#         for cov in cov_regs:
#             for edim in encode_dims:
#                 errors = results_overall[env][cov][edim]
#                 if errors:
#                     overall_means[cov].append(np.mean(errors))
#                     overall_stds[cov].append(np.std(errors))
#                 else:
#                     overall_means[cov].append(np.nan)
#                     overall_stds[cov].append(np.nan)
#         x_vals = np.array(encode_dims, dtype=float)
        
#         plt.figure(figsize=(8, 6))
#         plt.xscale('log')
#         plt.yscale('log')
#         for cov in cov_regs:
#             label = "Covariance Loss Off" if cov == 0 else "Covariance Loss On"
#             y = np.array(overall_means[cov])
#             y_std = np.array(overall_stds[cov])
#             plt.plot(x_vals, y, marker='o', label=label)
#             plt.fill_between(x_vals, y - y_std, y + y_std, alpha=0.3)
#         plt.xlabel("Encode Dimension")
#         plt.ylabel("Overall Multi-step Prediction Error")
#         plt.title(f"{env} - Overall Prediction Error")
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(os.path.join(fig_dir, f"{env}_overall_error.png"))
#         plt.close()
        
#         # Prepare data for Set B: Normalized covariance loss vs encode dimension.
#         norm_cov_means = {cov: [] for cov in cov_regs}
#         norm_cov_stds = {cov: [] for cov in cov_regs}
#         for cov in cov_regs:
#             for edim in encode_dims:
#                 cov_losses = results_cov[env][cov][edim]
#                 # Normalize by factor = enc_out_dim * (enc_out_dim - 1)
#                 norm_losses = []
#                 for cl in cov_losses:
#                     enc_dim = encode_dims_out[env][cov].get(edim, None)
#                     if enc_dim is None or enc_dim <= 1:
#                         norm_losses.append(cl)
#                     else:
#                         norm_losses.append(cl / (enc_dim * (enc_dim - 1)))
#                 if norm_losses:
#                     norm_cov_means[cov].append(np.mean(norm_losses))
#                     norm_cov_stds[cov].append(np.std(norm_losses))
#                 else:
#                     norm_cov_means[cov].append(np.nan)
#                     norm_cov_stds[cov].append(np.nan)
        
#         plt.figure(figsize=(8, 6))
#         plt.xscale('log')
#         plt.yscale('log')
#         for cov in cov_regs:
#             label = "Covariance Loss Off" if cov == 0 else "Covariance Loss On"
#             y = np.array(norm_cov_means[cov])
#             y_std = np.array(norm_cov_stds[cov])
#             plt.plot(x_vals, y, marker='o', label=label)
#             plt.fill_between(x_vals, y - y_std, y + y_std, alpha=0.3)
#         plt.xlabel("Encode Dimension")
#         plt.ylabel("Normalized Covariance Loss")
#         plt.title(f"{env} - Normalized Covariance Loss")
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(os.path.join(fig_dir, f"{env}_normalized_cov_loss.png"))
#         plt.close()
        
#         # Prepare data for Set C: Per-step prediction error curves.
#         # For each combination (encode dimension, cov_reg), average per-step errors over seeds.
#         plt.figure(figsize=(10, 6))
#         plt.yscale('log')
#         for i, edim in enumerate(encode_dims):
#             color = cmap(color_indices[i])
#             for cov in cov_regs:
#                 all_curves = results_per_step[env][cov][edim]
#                 if not all_curves:
#                     continue
#                 # all_curves is a list of lists; convert to a 2D array: (num_seeds, num_steps)
#                 curves_array = np.array(all_curves)
#                 mean_curve = np.mean(curves_array, axis=0)
#                 std_curve = np.std(curves_array, axis=0)
#                 steps = np.arange(1, len(mean_curve) + 1)
#                 # Set linestyle: solid for cov=0, dashed for cov=1.
#                 linestyle = '-' if cov == 0 else '--'
#                 label = f"edim={edim}, {'No CL' if cov == 0 else 'CL'}"
#                 plt.plot(steps, mean_curve, color=color, linestyle=linestyle, label=label)
#                 plt.fill_between(steps, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.3)
#         plt.xlabel("Step")
#         plt.ylabel("Prediction Error")
#         plt.title(f"{env} - Per-step Prediction Error")
#         plt.legend(fontsize='small', ncol=2)
#         plt.tight_layout()
#         plt.savefig(os.path.join(fig_dir, f"{env}_per_step_error.png"))
#         plt.close()

# if __name__ == "__main__":
#     main()

import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../scripts')

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
    project_name = "Koopman_Results_Apr_7_1"
    model_dir = f"../log/best_models/{project_name}/"
    fig_dir = f"{project_name}_figure"
    os.makedirs(fig_dir, exist_ok=True)

    envs = ['Polynomial', 'LogisticMap', 'DampingPendulum', 'DoublePendulum', 'Franka', 'G1', 'Go2']  # extend as needed
    random_seeds = [1, 2, 3]
    encode_dims = [4, 16, 64, 256, 1024]
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
        collector = KoopmanDatasetCollector(env, train_samples, val_samples, test_samples, Ksteps, device, normalize=normalize)
        test_data = torch.from_numpy(collector.test_data).double()
        #test_data = test_data[:12, :, :]  # only first 12 steps
        state_dim = collector.state_dim
        u_dim = collector.u_dim

        for seed in random_seeds:
            for cov_reg in cov_regs:
                for edim in encode_dims:
                    fname = os.path.join(model_dir, f"best_model_{norm_str}_{env}_{edim}_{cov_reg}_{seed}.pth")
                    if not os.path.exists(fname):
                        print("  ✗", fname)
                        continue
                    saved = torch.load(fname, map_location=device)
                    layers = saved['layer']
                    Nkoopman = state_dim + layers[-1]
                    model = KoopmanNet(layers, Nkoopman, u_dim)
                    model.load_state_dict(saved['model'])
                    model.to(device).double().eval()

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
