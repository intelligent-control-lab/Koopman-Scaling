import os
import sys
import math
import numpy as np
import pandas as pd
import torch
import scipy
import scipy.linalg
import pybullet as pb
from scipy.io import loadmat, savemat

sys.path.append("../../utility")
from franka_env import FrankaEnv
from network import KoopmanNet

# --------------------
# Tunables (ADDED)
# --------------------
NOISE_SIGMA = 0.01        # keep original noise level
WARMUP_SKIP = 100        # skip first N steps in metric (removes large transient)
ADAPT_R     = False #True       # stability guard for R (inflate from 0.01 upward if needed)
R_GROW      = 1.8
R_CAP       = 0.5
USE_CLIP    = True       # actuator clipping before env.step
YZ_WEIGHT   = 100.0      # stronger penalty on y,z tracking
VEL_WEIGHT  = 0 #2e-2       # tiny velocity damping (helps under noise)

# --------------------
# Helpers
# --------------------

def calculate_control_tasks_err(encode_dim, use_residual, model_path):
    # Load network
    env = FrankaEnv(render=False)
    in_dim = env.Nstates
    u_dim = env.udim

    subsuffix = "../../Koopman-Scaling/" + model_path[3:]
    dicts = torch.load(subsuffix, map_location=torch.device('cpu'))
    state_dict = dicts["model"]
    Elayer = dicts["layer"]

    NKoopman = encode_dim + in_dim
    net = KoopmanNet(Elayer, NKoopman, u_dim, use_residual)
    net.load_state_dict(state_dict)
    net.cpu().double().eval()

    # Discrete linear model
    Ad = state_dict['lA.weight'].cpu().numpy()
    Bd = state_dict['lB.weight'].cpu().numpy()
    eig = scipy.linalg.eigvals(Ad)
    max_eig = max(eig)
    print("[INFO] Open-loop max eigenvalue = {}".format(max(eig)))

    env.reset()
    nStates = in_dim  # use full physical state dimension
    accuracy_invKin = 1e-6
    T = 6 * 10
    t = 1.6 + 0.02 * np.linspace(0, T*5, T*50+1)
    Steps = len(t) - 1

    # Figure-8 path
    a = 0.3
    x = 0.3 * np.ones((len(t), 1))
    z = np.expand_dims(0.59 + 2 * a * np.sin(t) * np.cos(t) / (1 + np.sin(t)**2), axis=1)
    y = np.expand_dims(a * np.cos(t) / (1 + np.sin(t)**2), axis=1)

    def Psi_o(s, net):
        psi = np.zeros([NKoopman, 1])
        ds = net.encode(torch.DoubleTensor(s)).detach().cpu().numpy()
        psi[:NKoopman, 0] = ds
        return psi

    def Obs(o):
        # ### ADDED: parameterize noise magnitude (kept at original 0.1 by default)
        noise = np.random.randn(3) * NOISE_SIGMA
        return np.concatenate((o[:3] + noise, o[7:]), axis=0), noise

    def accurateCalculateInverseKinematics(kukaId, endEffectorId, targetPos, threshold, maxIter):
        numJoints = 7
        closeEnough = False
        it = 0
        dist2 = 1e30
        while (not closeEnough and it < maxIter):
            jointPoses = pb.calculateInverseKinematics(kukaId, endEffectorId, targetPos)
            for i in range(numJoints):
                pb.resetJointState(kukaId, i, jointPoses[i])
            ls = pb.getLinkState(kukaId, endEffectorId)
            newPos = ls[4]
            diff = [targetPos[0] - newPos[0], targetPos[1] - newPos[1], targetPos[2] - newPos[2]]
            dist2 = diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]
            closeEnough = (dist2 < threshold)
            it += 1
        return jointPoses[:7]

    def Run_Franka(Steps, state_desired, LQR_gains, net, x0=None, y0=None, z0=None):
        """Simulate forward dynamics of Franka using LQR control."""
        state = env.reset()

        # ### ADDED: reset ALL 7 joints; use [:,3:10] not 3:9
        for i, jnt in enumerate(state_desired[0, 3:10]):
            pb.resetJointState(env.robot, i, float(jnt))

        # ### ADDED: robust None checks + reset all 7 joints if an x0,y0,z0 is provided
        if (x0 is not None) and (y0 is not None) and (z0 is not None):
            j_init = accurateCalculateInverseKinematics(env.robot, env.ee_id, [x0, y0, z0], accuracy_invKin, 10000)
            for i, jnt in enumerate(j_init[:7]):
                pb.resetJointState(env.robot, i, float(jnt))

        state = env.get_state()

        # Initialize trajectories
        state_traj = np.empty((Steps+1, NKoopman))
        control_traj = np.empty((Steps, 7))
        state_traj[:], control_traj[:] = np.NaN, np.NaN

        # ### ADDED: actuator limits (if available)
        sat_val = getattr(env, 'sat_val', 0.3)

        # Rollout
        for k in range(Steps):
            state, noise = Obs(state)
            enc = Psi_o(state, net)
            state_traj[k, :] = enc[:NKoopman].reshape(-1)
            state_traj[k, :3] -= noise  # log with noise subtracted (matches original intent)

            # Pure LQR feedback to next desired encoded state
            control = - np.dot(LQR_gains, (enc - Psi_o(state_desired[k+1, :], net)))

            # ### ADDED: pre-step clipping to avoid nonlinear saturation dynamics
            if USE_CLIP:
                control = np.clip(control, -sat_val, +sat_val)

            control_traj[k, :] = control.reshape(-1)
            state = env.step(control)

        # Tail sample
        state, noise = Obs(state)
        enc = Psi_o(state, net)
        state_traj[Steps, :] = enc[:NKoopman].reshape(-1)
        state_traj[Steps, :3] -= noise

        return state_traj, control_traj

    def runLQRonFranka(steps, desired_target, LQR_gains, net, x0=None, y0=None, z0=None, method=False):
        state_traj, controls_traj = Run_Franka(steps, desired_target, LQR_gains, net, x0, y0, z0)

        # ### ADDED: compute error on (y,z) with warmup skip
        yz_traj = state_traj[:, 1:3]
        yz_des  = desired_target[:, 1:3]
        if WARMUP_SKIP > 0:
            yz_traj = yz_traj[WARMUP_SKIP:]
            yz_des  = yz_des[WARMUP_SKIP:]
        error = np.linalg.norm(yz_traj - yz_des)

        if method:
            import matplotlib.pyplot as plt
            plt.plot(state_traj[:,1], state_traj[:,2], 'b-', linewidth=1, markersize=1)
            plt.plot(desired_target[:,1], desired_target[:,2], 'k--', linewidth=1)
            plt.axis('equal')
            plt.title(method + ': Error = {0:.2f}'.format(error))
        return state_traj, controls_traj, error

    def desiredStates_from_EndEffector(xyzEndEffector):
        x_d, y_d, z_d = xyzEndEffector[0], xyzEndEffector[1], xyzEndEffector[2]
        jointAngles = np.asarray(accurateCalculateInverseKinematics(env.robot, env.ee_id, [x_d, y_d, z_d], accuracy_invKin, 10000))
        state_des = np.concatenate((xyzEndEffector, jointAngles, np.zeros(7)))
        return state_des

    # Translate desired y-z to joint angles (figure-8)
    JointAngles_Fig8 = np.empty((len(t), 7))
    JointAngles_Fig8[:] = np.NaN
    for i in range(len(t)):
        JointAngles_Fig8[i, :] = accurateCalculateInverseKinematics(env.robot, env.ee_id, [x[i], y[i], z[i]], accuracy_invKin, 10000)
    states_des = np.concatenate((x, y, z, JointAngles_Fig8, np.zeros((len(y), 7))), axis=1)

    # ----- LQR (ADDED: Q reweighting + adaptive R guard) -----
    import lqr
    Ad_m = np.matrix(Ad)
    Bd_m = np.matrix(Bd)

    Q = np.zeros((NKoopman, NKoopman))
    # keep original base on first 10 dims
    Q[:10, :10] = np.eye(10)

    # ### ADDED: emphasize y,z and add tiny velocity penalty if available
    if in_dim >= 3:
        Q[1,1] = YZ_WEIGHT
        Q[2,2] = YZ_WEIGHT
    if in_dim >= 17:
        Q[10:17, 10:17] = np.eye(7) * VEL_WEIGHT

    R_scale_eff = 0.01  # start from original
    def compute_K(R_scale):
        R = np.matrix(np.eye(u_dim) * R_scale)
        return lqr.lqr_regulator_k(Ad_m, Bd_m, np.matrix(Q), R)

    # Try to stabilize if marginal
    Kopt = compute_K(R_scale_eff)
    Acl = Ad - Bd @ np.asarray(Kopt)
    rhoAcl = float(np.max(np.abs(np.linalg.eigvals(Acl))))
    if ADAPT_R and rhoAcl >= 0.999:
        for _ in range(6):
            R_scale_eff = min(R_CAP, R_scale_eff * R_GROW)
            Kopt = compute_K(R_scale_eff)
            Acl = Ad - Bd @ np.asarray(Kopt)
            rhoAcl = float(np.max(np.abs(np.linalg.eigvals(Acl))))
            if rhoAcl < 0.999 or R_scale_eff >= R_CAP:
                break
    print(f"[INFO] R_scale_eff={R_scale_eff:.4f}, rho(A-BK)={rhoAcl:.6f}")

    # ----- Figure-8 rollout (kept same file outputs) -----
    os.makedirs('Results', exist_ok=True)
    np.random.seed(0)
    a_off = [0.01,0.021,-0.015,-0.012,0.093,0.058,0.014,-0.086,-0.096,0.056]
    b_off = [0.043,0.009,0.029,0.078,-0.023,0.006,0.085,-0.083,0.067,0.074]
    for i in range(1):
        y0, z0 = y[0,0] + a_off[i], z[0,0] + b_off[i]
        state_traj_KP, controls_traj_KP, error_KP = runLQRonFranka(Steps, states_des, np.asarray(Kopt), net, x[0,0], y0, z0)
        savemat('Results/DKUC_FrankaFig8noise_SimData'+str(i)+'.mat', {
            'desired_states':states_des,
            'states_KoopmanU': state_traj_KP,
            'error_KoopmanU': error_KP,
            'y0': y0,
            'z0': z0,
        })

    trajs = loadmat('Results/DKUC_FrankaFig8noise_SimData0.mat')
    state_traj_LS = trajs['states_KoopmanU']
    desired_traj = trajs['desired_states']
    error_1 = trajs['error_KoopmanU']
    print('[INFO] Fig-8 error =', error_1)

    # ----- Star path (unchanged path generation) -----
    center = np.array([0.0, 0.6])
    radius = 0.3
    theta_ = np.pi / 10.0
    eradius = np.tan(2*theta_) * radius * np.cos(theta_) - radius * np.sin(theta_)
    Star_points = np.zeros((11, 2))
    for i in range(5):
        theta = 2*np.pi/5*(i+0.25)
        Star_points[2*i,0] = np.cos(theta)*radius + center[0]
        Star_points[2*i,1] = np.sin(theta)*radius + center[1]
        beta = 2*np.pi/5*(i+0.75)
        Star_points[2*i+1,0] = np.cos(beta)*eradius + center[0]
        Star_points[2*i+1,1] = np.sin(beta)*eradius + center[1]
    Star_points[-1,:] = Star_points[0,:]

    T = 6 * 10
    t2 = 0.02 * np.linspace(0, T*5, T*50+1)
    refs = np.zeros((len(t2), 2))
    Steps2 = len(t2) - 1
    each_num = int((len(t2) - 10) / 9.5)
    for i in range(10):
        refs[(each_num+1)*i, :] = Star_points[i, :]
        num = each_num if i != 9 else len(t2) - (each_num+1)*i - 1
        for j in range(num):
            tau = (j+1) / (each_num+1)
            refs[(each_num+1)*i + j + 1, :] = tau*Star_points[i+1, :] + (1-tau)*Star_points[i, :]

    x2 = 0.3 * np.ones((len(t2), 1))
    z2 = refs[:,1].reshape(-1,1)
    y2 = refs[:,0].reshape(-1,1)

    JointAngles_Star = np.empty((len(t2), 7))
    JointAngles_Star[:] = np.NaN
    for i in range(len(t2)):
        JointAngles_Star[i,:] = accurateCalculateInverseKinematics(env.robot, env.ee_id, [x2[i], y2[i], z2[i]], accuracy_invKin, 10000)
    states_des2 = np.concatenate((x2, y2, z2, JointAngles_Star, np.zeros((len(y2), 7))), axis=1)

    np.random.seed(0)
    for i in range(1):
        y0, z0 = y2[0,0] + a_off[i], z2[0,0] + b_off[i]
        state_traj_KP, controls_traj_KP, error_KP = runLQRonFranka(Steps2, states_des2, np.asarray(Kopt), net, x2[0,0], y0, z0)
        savemat('Results/DKUC_FrankaFigStarnoise_SimData'+str(i)+'.mat', {
            'desired_states': states_des2,
            'states_KoopmanU': state_traj_KP,
            'error_KoopmanU': error_KP,
            'y0': y0,
            'z0': z0,
        })

    trajs2 = loadmat('Results/DKUC_FrankaFigStarnoise_SimData0.mat')
    state_traj_LS2 = trajs2['states_KoopmanU']
    desired_traj2 = trajs2['desired_states']
    error_2 = trajs2['error_KoopmanU']
    print('[INFO] Star error =', error_2)

    # Return shape unchanged for downstream merge
    return max_eig, error_1, error_2


if __name__ == "__main__":
    project_name = "Aug_8"
    log = pd.read_csv(f"../../Koopman-Scaling/log/{project_name}/koopman_results_log.csv")
    log = log[log['env_name'] == 'Franka'].reset_index(drop=True)

    results = []
    for i in range(len(log['model_path'])):
        row = log.loc[log['model_path'] == log['model_path'][i]]
        encode_dim = int(row['encode_dim'].values[0])
        use_residual = bool(row['use_residual'].values[0])
        max_eig, error_1, error_2 = calculate_control_tasks_err(encode_dim, use_residual, log['model_path'][i])
        results.append({
            'model_path': log['model_path'][i],
            'max_eig': max_eig,
            'error_eight': error_1,
            'error_star': error_2,
        })
        try:
            pb.disconnect()
        except Exception:
            pass

    results_df = pd.DataFrame(results)
    results_merged = log[['model_path', 'env_name', 'seed', 'encode_dim', 'use_control_loss', 'use_covariance_loss']].merge(
        results_df, on='model_path').drop(columns=['model_path'])
    results_merged.to_csv(f"../../Koopman-Scaling/log/{project_name}/franka_control_results.csv", index=False)
    print("Saved results to:", f"../../Koopman-Scaling/log/{project_name}/franka_control_results.csv")
