# Koopman-Scaling

Neural Koopman operator experiments for the scaling-law study in our paper.

This repository includes:
- data collection/loading pipelines for each environment,
- model training and large hyperparameter sweeps,
- Isaac Lab MPC tracking evaluation for `G1` and `Go2`,
- notebooks to compute metrics and generate paper plots.

## Repository Structure

- `scripts/train_model.py`: main training entrypoint.
- `scripts/run_experiments.sh`: full hyperparameter sweep used for scaling-law studies.
- `scripts/run_corr_experiments.sh`: corrected `m = c * n * log(n)` sweep for `G1`/`Go2`.
- `utility/dataset.py`: all dataset collectors and train/val/test construction.
- `utility/network.py`: Koopman network architecture.
- `control/mpc_tracking.py`: batch Isaac Lab tracking evaluation for `G1`/`Go2`.
- `control/run_all_models.sh`: repeatedly calls tracking until all models are processed.
- `evaluation/evaluate_prediction.ipynb`: prediction metrics/plots.
- `evaluation/evaluate_tracking.ipynb`: tracking metrics/plots.
- `evaluation/evaluate_correlation.ipynb`: scaling/correlation analysis.
- `evaluation/evaluate_covariance.ipynb`: covariance-related analysis.

## Environments and Data Sources

`utility/dataset.py` (`KoopmanDatasetCollector`) supports:
- `Polynomial`: synthetic polynomial dynamics generated online (`PolynomialDataCollector`).
- `LogisticMap`: synthetic logistic map generated online (`LogisticMapDataCollector`).
- `DampingPendulum`: ODE rollouts with random torque (`DampingPendulumDataCollector`).
- `DoublePendulum`: ODE rollouts with random torques (`DoublePendulumDataCollector`).
- `Franka`: PyBullet rollouts of 7-DoF arm velocity control (`FrankaDataCollector`).
- `Kinova`: loaded from logged files in `../data/kinova_data/` (`KinovaDataCollector`).
- `G1`, `Go2`: loaded from pre-collected `.npz` datasets in `../data/g1_flat/` and `../data/unitree_go2_flat/` (`G1Go2DataCollector`).

For `G1`/`Go2`, raw robot states/actions are trimmed before training:
- `G1`: state dim `53`, action dim `23`.
- `Go2`: state dim `35`, action dim `12`.

Datasets are cached at:
- `../data/datasets/dataset_<env>_<norm|nonorm>_... .pt`

## Training Pipeline

Run from `scripts/` (important because paths in code use `../...`):

```bash
cd scripts
python train_model.py \
  --project_name Sep_21 \
  --env_name G1 \
  --sample_size 64000 \
  --encode_dim 4 \
  --layer_depth 3 \
  --hidden_dim 256 \
  --seed 17382 \
  --m 0 \
  --use_residual \
  --use_control_loss \
  --use_covariance_loss \
  --multiply_encode_by_input_dim
```

Main behavior (`scripts/train_model.py`):
- Builds Koopman encoder + linear latent dynamics (`A`, `B`).
- Trains with multi-step Koopman prediction loss.
- Optionally adds control reconstruction loss and covariance regularization.
- Uses env-specific defaults:
  - `Ksteps=15` for most envs, `Ksteps=1` for `Polynomial`/`LogisticMap`.
  - normalization enabled for `G1`/`Go2`, disabled otherwise.
- Saves best checkpoint and appends one row to experiment CSV log.

Outputs:
- models: `../log/<project_name>/best_models/<timestamp>_model_<env>.pth`
- summary CSV: `../log/<project_name>/koopman_results_log.csv`

## Large-Scale Sweeps

From `scripts/`:

```bash
bash run_experiments.sh
```

This script:
- enumerates envs/seeds/encode dims/sample sizes/loss toggles,
- skips completed runs by checking `koopman_results_log.csv`,
- launches `train_model.py` for missing combinations only.

Corrected scaling-law sweep (`G1`/`Go2` only):

```bash
bash run_corr_experiments.sh
```

In this script, sample size is computed as:
- `m = coeff * n_eff * ln(n_eff)`
- `n_eff = encode_dim_multiplier * state_dim(env)`

## G1/Go2 Tracking Evaluation (Isaac Lab MPC)

`control/mpc_tracking.py` evaluates trained models by running Koopman-MPC tracking in Isaac Lab and writing per-model metrics.

Inputs:
- model list from `../log/<project>/koopman_results_log.csv`
- reference trajectories from `../data/g1_flat/reference_repository/` and `../data/unitree_go2_flat/reference_repository/`

Run from `control/`:

```bash
cd control
python mpc_tracking.py --headless \
  --csv_log_path ../log/Sep_21/koopman_results_log.csv \
  --save_path ../log/Sep_21/isaac_control_results.csv
```

Or process in resume loop:

```bash
bash run_all_models.sh
```

Tracking outputs:
- `../log/<project>/isaac_control_results.csv`
- Metrics include `mean_JrPE`, `mean_JrVE`, `mean_JrAE`, root errors, survival steps, and runtime.

## Paper Plots and Metrics

Use notebooks in `evaluation/` to compute and plot final paper results:
- `evaluate_prediction.ipynb`: prediction performance summaries/plots.
- `evaluate_tracking.ipynb`: tracking performance plots from Isaac MPC results.
- `evaluate_correlation.ipynb`: correlation/scaling-law visualizations.
- `evaluate_covariance.ipynb`: covariance-term analyses.

## Notes

- Required datasets are expected under `../data/...` relative to `scripts/` and `control/`.
- Logs/checkpoints are written under `../log/...`.
- For consistent paths, run scripts from their own directories (`scripts/`, `control/`).
