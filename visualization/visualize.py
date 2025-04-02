import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools
import sys
import argparse
sys.path.append('../scripts')

from dataset import KoopmanDatasetCollector
from network import KoopmanNet
from train import cov_loss

def compute_multistep_error(net, test_data, u_dim, gamma, device="cpu"):
    """Compute multi-step error using forward predictions."""
    mse_per_step = []
    net.eval()
    with torch.no_grad():
        steps = test_data.shape[0]
        data_dim = test_data.shape[2]
        state_dim = data_dim if u_dim is None else data_dim - u_dim
        test_data = torch.tensor(test_data, dtype=torch.double, device=device)
        X_current = net.encode(test_data[0, :, u_dim:] if u_dim else test_data[0, :])
        for i in range(1, steps):
            X_current = net.forward(X_current, test_data[i-1, :, :u_dim] if u_dim is not None else None)
            pred_state = X_current[:, :state_dim]
            true_state = test_data[i, :, u_dim:] if u_dim else test_data[i, :]
            mse_per_step.append(torch.mean((pred_state - true_state) ** 2).item())
        final_error = mse_per_step[-1] if mse_per_step else None
    return mse_per_step, final_error

def evaluate_on_test(net, test_data, device="cpu", gamma=0.8, u_dim=None):
    """Evaluate network on test data computing covariance loss and final-step MSE."""
    net.eval()
    test_tensor = torch.tensor(test_data, dtype=torch.double, device=device)
    state_dim = test_tensor.shape[2] if u_dim is None else test_tensor.shape[2] - u_dim
    with torch.no_grad():
        init_state = test_tensor[0, :, u_dim:] if u_dim is not None else test_tensor[0, :, :]
        initial_encoding = net.encode(init_state)
        c_loss = cov_loss(initial_encoding).item()
    n_steps = test_tensor.shape[0] - 1
    X_current = net.encode(test_tensor[0, :, u_dim:] if u_dim else test_tensor[0, :])
    final_mse = None
    with torch.no_grad():
        for i in range(n_steps):
            X_current = net.forward(X_current, test_tensor[i, :, :u_dim] if u_dim is not None else None)
            pred_state = X_current[:, :state_dim]
            true_state = test_tensor[i+1, :, u_dim:] if u_dim else test_tensor[i+1, :]
            final_mse = torch.mean((pred_state - true_state)**2).item()
    return final_mse, c_loss

def main():
    parser = argparse.ArgumentParser(description='Visualize KoopmanNet results.')
    parser.add_argument('--envs', nargs='+', default=['LogisticMap', 'DampingPendulum', 'DoublePendulum', 'Franka', 'Polynomial', 'G1', 'Go2'])
    parser.add_argument('--log-scale', type=lambda x: (str(x).lower() == 'true'), default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ENVIRONMENTS = args.envs
    USE_LOG_SCALE = args.log_scale

    ENCODE_DIMS = [1, 4, 16, 64, 256, 512, 1024]
    COV_REGS = [0, 1]
    SEEDS = [1]
    MODEL_DIR = "../log/best_models"
    GAMMA = 0.8
    NORMALIZE = True

    output_folder = "figures"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    results = {}

    for env, dim, cov, seed in itertools.product(ENVIRONMENTS, ENCODE_DIMS, COV_REGS, SEEDS):
        if env == "Franka" or env == "LogisticMap":
            NORMALIZE = False
        else:
            NORMALIZE = True
        norm_str = "norm" if NORMALIZE else "nonorm"
        model_fname = f"best_model_{norm_str}_{env}_{dim}_{cov}_{seed}.pth"
        model_path = os.path.join(MODEL_DIR, model_fname)
        if not os.path.exists(model_path):
            print(f"[WARNING] No saved model found: {model_path}")
            continue

        saved_dict = torch.load(model_path, map_location=device)
        best_state = saved_dict['model']
        layer_config = saved_dict['layer']

        Ksteps = 1 if env in ["Polynomial", "LogisticMap"] else 15

        collector = KoopmanDatasetCollector(env_name=env, 
                                              train_samples=60000,
                                              val_samples=20000,
                                              test_samples=20000,
                                              Ksteps=Ksteps,
                                              device=device)
        _, _, test_data = collector.get_data()
        u_dim = collector.u_dim
        state_dim = collector.state_dim
        Nkoopman = state_dim + layer_config[-1]

        net = KoopmanNet(layer_config, Nkoopman, u_dim).to(device).double()
        net.load_state_dict(best_state)
        net.eval()

        stepwise_mse, final_mse_multistep = compute_multistep_error(net, test_data, u_dim, GAMMA, device)
        final_mse, final_cov = evaluate_on_test(net, test_data, device, GAMMA, u_dim)

        results.setdefault(env, {}).setdefault(dim, {}).setdefault(cov, {})[seed] = {
            "final_mse_multistep": final_mse_multistep,
            "stepwise_mse": stepwise_mse,
            "final_mse": final_mse,
            "final_cov": final_cov/dim
        }

    for env in ENVIRONMENTS:
        if env not in results:
            continue

        dims_available = sorted(results[env].keys())
        final_means_cov0, final_stds_cov0 = [], []
        final_means_cov1, final_stds_cov1 = [], []

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
                mean_ms, std_ms = np.mean(all_mses), np.std(all_mses)
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
        if USE_LOG_SCALE:
            plt.yscale("log")
        plt.savefig(os.path.join(output_folder, f"{env}_finalMSE_vs_dim.png"), dpi=300, bbox_inches="tight")
        plt.close()

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
        if USE_LOG_SCALE:
            plt.yscale("log")
        plt.savefig(os.path.join(output_folder, f"{env}_bar_cov_on_off.png"), dpi=300, bbox_inches="tight")
        plt.close()

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
            if USE_LOG_SCALE:
                plt.yscale("log")
            plt.savefig(os.path.join(output_folder, f"{env}_multistep_cov{cov}.png"), dpi=300, bbox_inches="tight")
            plt.close()

        mse_means_cov0, mse_stds_cov0 = [], []
        cov_means_cov0, cov_stds_cov0 = [], []
        mse_means_cov1, mse_stds_cov1 = [], []
        cov_means_cov1, cov_stds_cov1 = [], []
        for d in dims_available:
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
        ax2.errorbar(xvals, cov_means_cov0, yerr=cov_stds_cov0, marker='^', linestyle='--', capsize=3, label="CovLoss (Cov=Off)")
        ax2.errorbar(xvals, cov_means_cov1, yerr=cov_stds_cov1, marker='^', linestyle='--', capsize=3, label="CovLoss (Cov=On)")
        ax2.set_ylabel("Covariance Loss")
        if USE_LOG_SCALE:
            ax1.set_yscale("log")
            ax2.set_yscale("log")
        else:
            ax1.set_yscale("linear")
            ax2.set_yscale("linear")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')
        plt.title(f"{env} - MSE & CovLoss vs. Encoding Dimension")
        plt.savefig(os.path.join(output_folder, f"{env}_mse_covloss_vs_dim.png"), dpi=300, bbox_inches="tight")
        plt.close()

    print("All plots generated successfully!")

if __name__ == "__main__":
    main()