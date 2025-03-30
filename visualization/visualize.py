import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools
import sys
sys.path.append('../scripts')

from dataset import KoopmanDatasetCollector
from network import KoopmanNet
from train import cov_loss


# --------------------------------------------------------------------
# 1) Helper Function: Compute Multi-step Error (from Script 1)
# --------------------------------------------------------------------
def compute_multistep_error(net, test_data, u_dim, gamma, device="cpu"):
    """
    Computes the MSE at each prediction step using multi-step forward predictions.
    
    Args:
        net: Trained KoopmanNet.
        test_data: Array of shape (Ksteps+1, nTraj, data_dim) (if u_dim is None) or 
                   (Ksteps+1, nTraj, state_dim+u_dim) if controls exist.
        u_dim: Control dimension (or None if no control).
        gamma: Discount factor used in training (for consistency).
        device: 'cpu' or 'cuda'.
    
    Returns:
       mse_per_step: List of MSE values at each prediction step.
       final_error: MSE at the final prediction step.
    """
    mse_per_step = []
    net.eval()

    with torch.no_grad():
        steps = test_data.shape[0]
        nTraj = test_data.shape[1]
        data_dim = test_data.shape[2]

        if u_dim is None:
            state_dim = data_dim
        else:
            state_dim = data_dim - u_dim

        test_data = torch.tensor(test_data, dtype=torch.double, device=device)

        # Initial encoding: if control exists, remove control components from initial time
        X_current = net.encode(test_data[0, :, u_dim:] if u_dim else test_data[0, :])
        for i in range(1, steps):
            if u_dim is not None:
                X_current = net.forward(X_current, test_data[i-1, :, :u_dim])
            else:
                X_current = net.forward(X_current, None)

            pred_state = X_current[:, :state_dim]
            true_state = test_data[i, :, u_dim:] if u_dim else test_data[i, :]
            mse_i = torch.mean((pred_state - true_state) ** 2).item()
            mse_per_step.append(mse_i)

        final_error = mse_per_step[-1] if mse_per_step else None

    return mse_per_step, final_error

# --------------------------------------------------------------------
# 2) Helper Function: Evaluate on Test (from Script 2)
# --------------------------------------------------------------------
def evaluate_on_test(net, test_data, device="cpu", gamma=0.8, u_dim=None):
    """
    Evaluates a network on test data by computing:
       A) Covariance loss on the initial encoding.
       B) Final-step MSE over multi-step prediction.
    
    Args:
        net: A KoopmanNet with .encode() and .forward() methods.
        test_data: Array of shape (Ksteps+1, nTraj, data_dim) where data_dim is 
                   state_dim+u_dim if controls exist.
        device: 'cpu' or 'cuda'.
        gamma: Discount factor (for consistency with training).
        u_dim: Dimension of control (if any).
    
    Returns:
        final_mse: Final step MSE.
        c_loss: Covariance loss computed on the initial encoding.
    """
    net.eval()
    test_tensor = torch.tensor(test_data, dtype=torch.double, device=device)

    if u_dim is None:
        state_dim = test_tensor.shape[2]
    else:
        state_dim = test_tensor.shape[2] - u_dim

    # A) Compute covariance loss on the initial encoding
    with torch.no_grad():
        if u_dim is not None:
            init_state = test_tensor[0, :, u_dim:]
        else:
            init_state = test_tensor[0, :, :]
        initial_encoding = net.encode(init_state)
        c_loss = cov_loss(initial_encoding).item()

    # B) Compute final-step MSE using multi-step prediction
    n_steps = test_tensor.shape[0] - 1
    X_current = net.encode(test_tensor[0, :, u_dim:] if u_dim else test_tensor[0, :])
    final_mse = None
    with torch.no_grad():
        for i in range(n_steps):
            if u_dim is not None:
                ctrl = test_tensor[i, :, :u_dim]
                X_current = net.forward(X_current, ctrl)
            else:
                X_current = net.forward(X_current, None)

            pred_state = X_current[:, :state_dim]
            true_state = test_tensor[i+1, :, u_dim:] if u_dim else test_tensor[i+1, :]
            final_mse = torch.mean((pred_state - true_state)**2).item()

    return final_mse, c_loss

# --------------------------------------------------------------------
# 3) Main Function: Load Models, Evaluate, and Generate Plots
# --------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configuration settings (common to both scripts)
    ENVIRONMENTS = ['G1']#['G1', 'Go2', 'LogisticMap', 'DampingPendulum', 'DoublePendulum', 'Franka', 'Polynomial']
    ENCODE_DIMS = [1, 4, 16, 64, 256, 512, 1024]
    COV_REGS = [0, 1]  # 0 => cov reg off, 1 => cov reg on
    SEEDS = [1]
    MODEL_DIR = "../log/best_models"  # where your best models are saved
    GAMMA = 0.8  # discount factor matching training
    results = {}  # dictionary to store evaluation metrics

    # Loop over all configurations
    for env, dim, cov, seed in itertools.product(ENVIRONMENTS, ENCODE_DIMS, COV_REGS, SEEDS):
        model_fname = f"best_model_{env}_{dim}_{cov}_{seed}.pth"
        model_path = os.path.join(MODEL_DIR, model_fname)
        if not os.path.exists(model_path):
            print(f"[WARNING] No saved model found: {model_path}")
            continue

        # Load saved model and layer configuration
        saved_dict = torch.load(model_path, map_location=device)
        best_state = saved_dict['model']
        layer_config = saved_dict['layer']

        # Load dataset (only test set needed)
        collector = KoopmanDatasetCollector(env_name=env, 
                                              train_samples=60000,
                                              val_samples=20000,
                                              test_samples=20000,
                                              Ksteps=15,
                                              device=device)
        _, _, test_data = collector.get_data()
        u_dim = collector.u_dim
        state_dim = collector.state_dim
        Nkoopman = state_dim + layer_config[-1]

        # Rebuild the network
        net = KoopmanNet(layer_config, Nkoopman, u_dim).to(device).double()
        net.load_state_dict(best_state)
        net.eval()

        # Compute multi-step errors (Script 1 evaluation)
        stepwise_mse, final_mse_multistep = compute_multistep_error(net, test_data, u_dim, GAMMA, device)

        # Compute final MSE and covariance loss (Script 2 evaluation)
        final_mse, final_cov = evaluate_on_test(net, test_data, device, GAMMA, u_dim)

        # Store results for this configuration
        if env not in results:
            results[env] = {}
        if dim not in results[env]:
            results[env][dim] = {}
        if cov not in results[env][dim]:
            results[env][dim][cov] = {}
        results[env][dim][cov][seed] = {
            "final_mse_multistep": final_mse_multistep,
            "stepwise_mse": stepwise_mse,
            "final_mse": final_mse,
            "final_cov": final_cov/dim
        }

    # ----------------------------------------------------------------
    # Plotting: Generate multiple plots for each environment.
    # ----------------------------------------------------------------
    for env in ENVIRONMENTS:
        if env not in results:
            continue

        dims_available = sorted(results[env].keys())
        
        # ---- Plot A: Line Plot of Final MSE (Multi-step) vs. Encoding Dimension ----
        final_means_cov0 = []
        final_stds_cov0 = []
        final_means_cov1 = []
        final_stds_cov1 = []

        for dim in dims_available:
            for cov in [0, 1]:
                if cov not in results[env][dim]:
                    if cov == 0:
                        final_means_cov0.append(np.nan)
                        final_stds_cov0.append(0)
                    else:
                        final_means_cov1.append(np.nan)
                        final_stds_cov1.append(0)
                    continue
                all_mses = [results[env][dim][cov][seed]["final_mse_multistep"] for seed in results[env][dim][cov]]
                mean_ms = np.mean(all_mses)
                std_ms = np.std(all_mses)
                if cov == 0:
                    final_means_cov0.append(mean_ms)
                    final_stds_cov0.append(std_ms)
                else:
                    final_means_cov1.append(mean_ms)
                    final_stds_cov1.append(std_ms)

        plt.figure()
        xvals = dims_available
        plt.errorbar(xvals, final_means_cov0, yerr=final_stds_cov0, marker='o', linestyle='-', capsize=3, label="Cov=Off")
        plt.errorbar(xvals, final_means_cov1, yerr=final_stds_cov1, marker='o', linestyle='-', capsize=3, label="Cov=On")
        plt.xlabel("Encoding Dimension")
        plt.ylabel("Final Test MSE (Multi-step)")
        plt.title(f"{env}: Final Prediction Error vs. Dimension")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{env}_finalMSE_vs_dim.png", dpi=300, bbox_inches="tight")
        plt.close()

        # ---- Plot B: Bar Chart Comparing Cov Regularization (Multi-step Final MSE) ----
        indices = np.arange(len(dims_available))
        width = 0.35

        plt.figure()
        plt.bar(indices - width/2, final_means_cov0, width, yerr=final_stds_cov0, label="Cov=Off")
        plt.bar(indices + width/2, final_means_cov1, width, yerr=final_stds_cov1, label="Cov=On")
        plt.xticks(indices, dims_available)
        plt.xlabel("Encoding Dimension")
        plt.ylabel("Final Test MSE (Multi-step)")
        plt.title(f"{env}: Cov Reg Comparison (Final MSE)")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{env}_bar_cov_on_off.png", dpi=300, bbox_inches="tight")
        plt.close()

        # ---- Plot C: Multi-step Error vs. Prediction Step for Each Cov Setting ----
        for cov in [0, 1]:
            plt.figure()
            for dim in dims_available:
                if cov not in results[env][dim]:
                    continue
                all_stepwise = [results[env][dim][cov][seed]["stepwise_mse"] for seed in results[env][dim][cov]]
                all_stepwise = np.array(all_stepwise)
                mean_stepwise = np.mean(all_stepwise, axis=0)
                std_stepwise = np.std(all_stepwise, axis=0)
                steps_plot = np.arange(1, len(mean_stepwise)+1)
                plt.errorbar(steps_plot, mean_stepwise, yerr=std_stepwise, marker='o', linestyle='-', capsize=3, label=f"Dim={dim}")
            plt.xlabel("Prediction Step")
            plt.ylabel("MSE")
            plt.title(f"{env}: Multi-step Error (Cov={cov})")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{env}_multistep_cov{cov}.png", dpi=300, bbox_inches="tight")
            plt.close()

        # ---- Plot D: Dual-Axis Plot for Final MSE and CovLoss (from evaluate_on_test) ----
        mse_means_cov0, mse_stds_cov0 = [], []
        cov_means_cov0, cov_stds_cov0 = [], []
        mse_means_cov1, mse_stds_cov1 = [], []
        cov_means_cov1 = []; cov_stds_cov1 = []
        for d in dims_available:
            # For cov=0
            if 0 in results[env][d]:
                mses = [results[env][d][0][seed]["final_mse"] for seed in results[env][d][0]]
                covs = [results[env][d][0][seed]["final_cov"] for seed in results[env][d][0]]
                mse_means_cov0.append(np.mean(mses))
                mse_stds_cov0.append(np.std(mses))
                cov_means_cov0.append(np.mean(covs))
                cov_stds_cov0.append(np.std(covs))
            else:
                mse_means_cov0.append(np.nan)
                mse_stds_cov0.append(0)
                cov_means_cov0.append(np.nan)
                cov_stds_cov0.append(0)
            # For cov=1
            if 1 in results[env][d]:
                mses = [results[env][d][1][seed]["final_mse"] for seed in results[env][d][1]]
                covs = [results[env][d][1][seed]["final_cov"] for seed in results[env][d][1]]
                mse_means_cov1.append(np.mean(mses))
                mse_stds_cov1.append(np.std(mses))
                cov_means_cov1.append(np.mean(covs))
                cov_stds_cov1.append(np.std(covs))
            else:
                mse_means_cov1.append(np.nan)
                mse_stds_cov1.append(0)
                cov_means_cov1.append(np.nan)
                cov_stds_cov1.append(0)

        fig, ax1 = plt.subplots()
        xvals = dims_available
        ax1.errorbar(xvals, mse_means_cov0, yerr=mse_stds_cov0, marker='o', linestyle='-', capsize=3, label="MSE (Cov=Off)")
        ax1.errorbar(xvals, mse_means_cov1, yerr=mse_stds_cov1, marker='o', linestyle='-', capsize=3, label="MSE (Cov=On)")
        ax1.set_xlabel("Encoding Dimension")
        ax1.set_ylabel("Final MSE")
        ax1.grid(True)
        ax2 = ax1.twinx()
        ax2.errorbar(xvals, cov_means_cov0, yerr=cov_stds_cov0, marker='^', linestyle='--', capsize=3, label="CovLoss (Cov=Off)", color="tab:blue")
        ax2.errorbar(xvals, cov_means_cov1, yerr=cov_stds_cov1, marker='^', linestyle='--', capsize=3, label="CovLoss (Cov=On)", color="tab:orange")
        ax2.set_ylabel("Covariance Loss")
        ax2.set_yscale("log")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')
        plt.title(f"{env} - MSE & CovLoss vs. Encoding Dimension")
        plt.savefig(f"{env}_mse_covloss_vs_dim.png", dpi=300, bbox_inches="tight")
        plt.close()

    print("All plots generated successfully!")

if __name__ == "__main__":
    main()