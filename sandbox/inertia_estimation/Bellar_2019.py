"""
Created by Elias Obreque
Date: 28-06-2025
email: els.obrq@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt

# EKF-based Inertia Matrix Estimation for CubeSat
# Incorporating inertia dynamics from Bellar & Si Mohammed (2019), Eq. (2)

# --- Helper functions ---
def inertia_matrix(params):
    Ixx, Iyy, Izz, Ixy, Ixz, Iyz = params
    return np.array([[Ixx, Ixy, Ixz],
                     [Ixy, Iyy, Iyz],
                     [Ixz, Iyz, Izz]])

# Dynamics include angular acceleration and inertia decay
def dynamics(state, tau_m, tau_p, t_ctrl):
    omega = state[:3]
    params = state[3:]
    I = inertia_matrix(params)
    # Euler's rotational dynamics
    domega = np.linalg.inv(I) @ (-np.cross(omega, I @ omega) + t_ctrl * np.array([1, 1, -2]) * 1e-3)
    # Inertia dynamics: moments (first 3) decay with tau_m, products (last 3) with tau_p
    Jm = params[:3]
    Jp = params[3:]
    if tau_m != 0 and tau_p != 0:
        dJm = -1.0/tau_m * Jm
        dJp = -1.0/tau_p * Jp
    else:
        dJm = 0.0 * Jm
        dJp = 0.0 * Jp

    dparams = np.hstack((dJm, dJp))
    return np.hstack((domega, dparams))

# Discrete propagation via RK4
def rk4_step(x, dt, tau_m, tau_p, t_ctrl=1):
    k1 = dynamics(x, tau_m, tau_p, t_ctrl)
    k2 = dynamics(x + 0.5*dt*k1, tau_m, tau_p, t_ctrl)
    k3 = dynamics(x + 0.5*dt*k2, tau_m, tau_p, t_ctrl)
    k4 = dynamics(x + dt*k3, tau_m, tau_p, t_ctrl)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# Jacobian of discrete map
def compute_discrete_F(x, dt, tau_m, tau_p, eps=1e-10):
    n = len(x)
    F = np.zeros((n,n))
    f0 = rk4_step(x, dt, tau_m, tau_p, t_ctrl=0.0)
    for i in range(n):
        dx = np.zeros(n);
        dx[i]=eps
        f1 = rk4_step(x+dx, dt, tau_m, tau_p, t_ctrl=0.0)
        F[:,i] = (f1 - f0)/eps
    return F

# Continuous-time Jacobian F with analytic inertia subblocks
def compute_F(state, tau_m, tau_p, t_ctrl=1):
    """
    Analytical Jacobian F(x):
      F = [[∂ω̇/∂ω, ∂ω̇/∂Jm, ∂ω̇/∂Jp],
           [0,     -1/τ_m*I,   0],
           [0,     0,       -1/τ_p*I]]
    """
    omega = state[:3]
    params = state[3:]
    I = inertia_matrix(params)
    invI = np.linalg.inv(I)
    # cross-product matrix
    Omega = np.array([[   0,     -omega[2],  omega[1]],
                      [omega[2],     0,    -omega[0]],
                      [-omega[1], omega[0],     0   ]])
    # ∂ω̇/∂ω = -I^{-1}( Omega*I + I*Omega )
    temp = I @ omega
    temp_cross = np.array([[0, -temp[2], temp[1]],
                      [temp[2], 0, -temp[0]],
                      [-temp[1], temp[0], 0]])
    F_ww = - invI @ (Omega @ I - temp_cross)
    # ∂ω̇/∂Jm: partial derivatives of domega wrt Ixx, Iyy, Izz
    # Each column j: ∂ω̇/∂Jm[j] = -invI * (E_j * domega) , where E_j selects param in inertia_matrix
    Tc = t_ctrl*np.array([1.0, 1.0, -2.0]) * 1e-3
    domega0 = Tc - Omega @ (I @ omega)
    F_wJm = np.zeros((3,3))
    for j, idx in enumerate([0,1,2]):
        E = np.zeros((3,3)); E[idx, idx] = 1
        d_invI = -invI @ E @ invI
        d_term = -invI @ (Omega @ (E @ omega))
        F_wJm[:, j] = d_invI @ domega0 + d_term
    # ∂ω̇/∂Jp for off-diagonals
    offidx = [(0,1),(0,2),(1,2)]
    F_wJp = np.zeros((3,3))
    for j,(r,c) in enumerate(offidx):
        E = np.zeros((3,3)); E[r,c] = E[c,r] = 1
        d_invI = -invI @ E @ invI
        d_term = -invI @ (Omega @ (E @ omega))
        F_wJp[:, j] = d_invI @ domega0 + d_term
    # Assemble full F
    F = np.zeros((9,9))
    F[0:3,0:3] = F_ww
    F[0:3,3:6] = F_wJm
    F[0:3,6:9] = F_wJp
    # inertia decay
    if tau_p != 0 and tau_m != 0:
        F[3:6,3:6] = -np.eye(3)/tau_m
        F[6:9,6:9] = -np.eye(3)/tau_p
    return F

# Continuous-time Jacobian F (analytic inertia subblocks + numeric elsewhere)
def compute_F_(state, tau_m, tau_p, eps=1e-6):
    n = len(state)
    F = np.zeros((n, n))
    # Numerical derivative for full Jacobian
    f0 = dynamics(state, tau_m, tau_p)
    for i in range(n):
        dx = np.zeros(n); dx[i] = eps
        F[:, i] = (dynamics(state + dx, tau_m, tau_p) - f0) / eps
    # Enforce analytical blocks per Eq.(7)
    # Inertia rows indices 3-5 (Jm) and 6-8 (Jp)
    # Zero those rows first
    F[3:6, :] = 0
    F[6:9, :] = 0
    # dJm/dJm = -1/tau_m * I3
    F[3:6, 3:6] = -np.eye(3) / tau_m
    # dJp/dJp = -1/tau_p * I3
    F[6:9, 6:9] = -np.eye(3) / tau_p
    return F

# Discrete transition matrix Phi via first-order Taylor (Eq.5)
def compute_Phi(state, dt, tau_m, tau_p):
    n = len(state)
    F = compute_F(state, tau_m, tau_p)
    return np.eye(n) + F * dt

# Measurement matrix: gyro measures only angular rates
H = np.hstack((np.eye(3), np.zeros((3,6))))

# --- Simulation parameters ---
dt = 0.1           # sample interval [s]
t_final = 600.0        # simulation time [s]
steps = int(t_final/dt) + 1
time = np.linspace(0, t_final, steps)

# True inertia (kg·m²) and initial angular rates (rad/s)
true_params = np.array([14.2, 17.3, 20.3, 0.0867, 0.1357, 0.6016])
# omega0 = np.deg2rad([0.0, -0.06, 0.0])
omega0 = np.array([-0.0001, 0.00110, 0.0001])
state_true = np.hstack((omega0, true_params))

# EKF initial guess (Table 2)
omega_est0 = np.deg2rad([5.0, -2.3, 1.15])
inertia_est0 = np.array([25.0,25.0,25.0,2.0,2.0,2.0])
x_est = np.hstack((omega_est0, inertia_est0))

# Covariances
gyro_std = np.deg2rad(0.0104)
P = np.diag(np.hstack(((gyro_std*10)**2 * np.ones(3),
                       (inertia_est0)**2)))

# Time-constants for inertia dynamics (Eq.2)
tau_m = 1e6   # for moments of inertia
tau_p = 1e6   # for products of inertia

# Process and measurement noise
# Qc_inertia = np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
# Q = np.diag(np.hstack((1e-6*np.ones(3), Qc_inertia))) * dt
R = (gyro_std**2) * np.eye(3)

# Storage
estimates = np.zeros((steps,9))
true_states = np.zeros((steps,9))
estimates[0] = x_est
true_states[0] = state_true

# Process noise spectral densities for angular rates (from paper, Eq.14)
sw = gyro_std**2 / dt
s_w = np.diag([sw]*3)

# --- Prepare noise storage ---
noise_series = np.zeros((steps, 3))
Ps = np.zeros((steps,9,9))
Ps[0] = P

# EKF loop
for k in range(1, steps):
    # propagate true and predicted state
    state_true = rk4_step(state_true, dt, 0, 0)
    true_states[k] = state_true

    x_pred = rk4_step(x_est, dt, tau_m, tau_p, t_ctrl=1)
    noise = gyro_std * np.random.randn(3)

    F = compute_F(x_est, tau_m, tau_p, t_ctrl=1)
    Phi = np.eye(9) + F * dt

    # Compute Q_k per Eq.(13)-(14)
    Fww = F[0:3, 0:3]
    dt2 = dt ** 2
    dt3 = dt ** 3
    term1 = s_w * dt
    term2 = s_w @ Fww.T * (dt2 / 2)
    term3 = Fww @ s_w * (dt2 / 2)
    term4 = Fww @ s_w @ Fww.T * (dt3 / 3)

    Qk = np.zeros((9, 9))
    Qk[0:3, 0:3] = term1 + term2 + term3 + term4

    P = Phi @ P @ Phi.T + Qk

    # Measurement
    y = true_states[k, :3] + noise

    # update
    reg = 1e-18
    S = H @ P @ H.T + R + reg * np.eye(3)
    K = P @ H.T @ np.linalg.inv(S)
    P = (np.eye(9) - K @ H) @ P
    x_est = x_pred + K @ (y - H @ x_pred)

    noise_series[k] = noise * (180 / np.pi)  # convert to deg/s

    Ps[k] = P
    estimates[k] = x_est

# --- Plot Inertia Convergence ---
labels_I = ['Ixx','Iyy','Izz','Ixy','Ixz','Iyz']
colors = ['r','g','b','c','m','y','k']
plt.figure(figsize=(10,6))
for i in range(6):
    plt.plot(time, estimates[:,3+i], color=colors[i], label=labels_I[i])
    plt.plot(time, true_states[:, 3+i], ls='--', color=colors[i], label=f'True {labels_I[i]}')
plt.xlabel('Time [s]')
plt.ylabel('Inertia [kg·m²]')
plt.title('EKF Convergence of Inertia Estimates')
plt.legend()
plt.tight_layout()

# --- Plot Angular Velocity: True vs Estimated ---
labels_w = ['ω_x','ω_y','ω_z']
fig, axes = plt.subplots(1, 3, figsize=(12, 5))
for i in range(3):
    axes[i].plot(time, true_states[:,i] * np.rad2deg(1), '-', label=rf'True ${labels_w[i]}$')
for i in range(3):
    axes[i].plot(time, estimates[:,i] * np.rad2deg(1), '--', label=rf'Est ${labels_w[i]}$')
    axes[i].set_xlabel('Time [s]')
    axes[i].legend()
axes[0].set_ylabel('Angular Rate [deg/s]')
fig.suptitle('True vs Estimated Angular Velocity')
plt.tight_layout()


##
# Convert to degrees
error_deg = (estimates[:, :3] - true_states[:, :3]) * (180 / np.pi)
sigma_omega = 3 * np.sqrt(Ps[:, :3, :3]) * (180 / np.pi)
labels_w = ['$ω_x$', '$ω_y$', '$ω_z$']

# Plot squared error per axis
fig2, axs2 = plt.subplots(3, 1, figsize=(8, 5), sharex=True)
for i in range(3):
    axs2[i].plot(time, error_deg[:, i]**2, 'b')
    axs2[i].set_ylabel(rf'{labels_w[i]} Error² (deg/s)²')
    axs2[i].set_xlim(0, t_final)
    axs2[i].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axs2[i].set_yscale('log')
axs2[-1].set_xlabel('Time [s]')
plt.suptitle('Squared Angular Velocity Estimation Error')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


omega_err = (estimates[:,:3] - true_states[:,:3])*(180/np.pi)
sigma_omega = 3 * np.sqrt([np.diag(m[:3, :3]) for m in Ps]) * (180/np.pi)

fig,axs=plt.subplots(3,2,figsize=(12,5), sharex=True)
for i in range(3):
    axs[i, 0].plot(time, noise_series[:, i], 'k', label='Noise')
    axs[i,0].plot(time, omega_err[:,i], 'r', label='Error')
    axs[i,0].set_ylabel('Error (deg/s)')
    axs[i,1].plot(time,omega_err[:,i],'r')
    axs[i,1].plot(time, sigma_omega[:,i],'k--')
    axs[i,1].plot(time,-sigma_omega[:,i],'k--')
    axs[i, 1].set_ylabel(rf'{labels_w[i]} Error (deg/s)')
    # axs[i, 0].set_yscale('log')
    #axs[i, 1].set_yscale('log')
plt.show()