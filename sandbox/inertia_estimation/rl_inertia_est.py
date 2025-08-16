"""
Created by Elias Obreque
Date: 05/08/2025
email: els.obrq@gmail.com
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduce Platanitis et al. (2024) – Inertial tensor identification via TSC + RL
Autor: R. Daneel Olivaw · Agosto 2025
Ejecuta tal cual; ajusta 'TOTAL_EPISODES' o 'NOISE_LEVELS' si lo necesitas.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
from gymnasium import spaces
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ---------- 1. Parámetros globales ----------
DT          = 0.2            # s paso integración RK4
SEQ_LEN     = 20             # nº de pulsos de la secuencia
TORQUE      = 2.0            # Nm nominal
PWM_LEVELS  = np.linspace(0, 1, 5)         # 5 niveles equiespaciados
NUM_THRUST  = 6
BASE_I      = np.diag([100, 60, 90])       # kg·m²
VAR_FRACT   = (0.05, 0.15)                 # rango de variación
N_MODELS    = 5                            # 1 base + 4 variantes
TOTAL_EPISODES = 3000                      # PPO training episodes
NOISE_LEVELS   = np.arange(0.5, 6.5, 0.5)  # para Fig. 2–3

# ---------- 2. Utilidades de dinámica ----------
def euler_rhs(omega, torque, I):
    return np.linalg.inv(I) @ (torque - np.cross(omega, I @ omega))

def rk4_step(omega, torque, I, dt):
    k1 = euler_rhs(omega, torque, I)
    k2 = euler_rhs(omega + 0.5*dt*k1, torque, I)
    k3 = euler_rhs(omega + 0.5*dt*k2, torque, I)
    k4 = euler_rhs(omega +    dt*k3,  torque, I)
    return omega + dt/6*(k1 + 2*k2 + 2*k3 + k4)

def build_inertias():
    rng = np.random.default_rng(42)
    inertias = [BASE_I]
    for _ in range(N_MODELS-1):
        delta = rng.uniform(VAR_FRACT[0], VAR_FRACT[1], size=3)
        inertias.append(np.diag((1-delta)*np.diag(BASE_I)))
    return inertias

# ---------- 3. Entorno Gym ----------
class SpacecraftEnv(gym.Env):
    '''Observación: secuencia de pulsos aplicada hasta ahora (longitud variable).
       Acción: vector PWM de 6 thrusters (discreto 5⁶, se usa codificación entera 0..4).'''
    metadata = {"render_modes": []}
    def __init__(self):
        super().__init__()
        self.inertias = build_inertias()
        self.actuation_levels = PWM_LEVELS
        # Acción discreta por thruster -> shape (6,)
        self.action_space = spaces.MultiDiscrete([len(self.actuation_levels)]*NUM_THRUST)
        # Observación = pulsos aplicados hasta ahora, padded a SEQ_LEN
        self.observation_space = spaces.Box(low=0, high=1, shape=(SEQ_LEN, NUM_THRUST), dtype=np.float32)
        self.reset(seed=0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.omega_hist = np.zeros((N_MODELS, SEQ_LEN+1, 3))
        self.act_hist   = np.zeros((SEQ_LEN, NUM_THRUST))
        self.omegas     = np.zeros((N_MODELS, 3))
        self.done       = False
        return self._get_obs(), {}

    def _apply_torque(self, act_vec, I):
        # Matriz A (Eq. 4) normalizada por TORQUE
        A = TORQUE * np.array([[ 1, 0, 0, -1, 0, 0],
                               [ 0, 1, 0,  0,-1, 0],
                               [ 0, 0, 1,  0, 0,-1]])
        return A @ act_vec

    def step(self, action):
        action = np.asarray(action, dtype=int)
        pwm = self.actuation_levels[action]           # map indices→duty cycles
        # low-pass emulación respuesta actuador (1ª orden)
        tau = 0.05
        pwm = (1-tau)*pwm + tau*np.random.normal(pwm, 0.02)
        for k,I in enumerate(self.inertias):
            torque = self._apply_torque(pwm, I)
            # ruido de proceso ~ N(0, 0.001 rad/s²)
            self.omegas[k] = rk4_step(self.omegas[k], torque,
                                      I, DT) + np.random.normal(0, 1e-3, size=3)
            self.omega_hist[k, self.step_count+1] = self.omegas[k]
        self.act_hist[self.step_count] = pwm
        self.step_count += 1
        reward, terminated = 0.0, False
        if self.step_count == SEQ_LEN:       # evaluar clasificador
            reward = self._evaluate_sequence()
            terminated = True
        obs = self._get_obs()
        return obs, reward, terminated, False, {}

    def _evaluate_sequence(self):
        # prepara dataset [n_models × seq_len × 3]
        X = self.omega_hist[:,1:, :]                      # quitar primer cero
        X = X / np.max(np.abs(X))                        # normaliza para DTW
        y_true = np.arange(N_MODELS)
        model = TimeSeriesKMeans(n_clusters=N_MODELS,
                                 metric="softdtw", max_iter=50,
                                 random_state=0)
        y_pred = model.fit_predict(X)
        f1 = f1_score(y_true, y_pred, average='macro')
        # reward Eq. (5) aproximado: -1+3 α  (no penalización ω̄ aquí)
        return -1 + 3*f1

    def _get_obs(self):
        obs = np.zeros((SEQ_LEN, NUM_THRUST))
        obs[:self.step_count] = self.act_hist[:self.step_count]
        return obs.astype(np.float32)

# ---------- 4. Entrenamiento PPO ----------
def train_ppo():
    env = SpacecraftEnv()
    check_env(env, warn=True)
    model = PPO("MlpPolicy", env,
                n_steps=2048,
                batch_size=64,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.05,
                vf_coef=0.5,
                learning_rate=3e-4,
                verbose=0)
    rewards = []
    for ep in range(TOTAL_EPISODES):
        obs, _ = env.reset()
        done, ep_rew = False, 0
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, r, done, _, _ = env.step(action)
            ep_rew += r
        model.learn(total_timesteps=SEQ_LEN, reset_num_timesteps=False)
        rewards.append(ep_rew)
        if (ep+1) % 50 == 0:
            print(f"Ep {ep+1}/{TOTAL_EPISODES}  MeanReward={np.mean(rewards[-50:]):.3f}")
    # Figura 1
    plt.figure(figsize=(6,3))
    plt.plot(rewards, lw=0.8)
    plt.title("Mean reward per episode (PPO)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig("fig_reward.png", dpi=200)
    return model, rewards

# ---------- 5. Robustness: ruido vs accuracy ----------
def evaluate_accuracy(model, noise_scales):
    env = SpacecraftEnv()
    base_proc_sigma = 1e-3
    base_meas_sigma = 1e-3
    acc_proc, acc_meas = [], []
    for mult in noise_scales:
        # proceso
        env = SpacecraftEnv()
        env.step = lambda a, env=env, mult=mult: SpacecraftEnv.step(
            env, a)[0:5]  # monkey patch not needed but kept minimal
        # Generate dataset with increased process noise
        acc_proc.append(_accuracy_single_run(mult, env, proc=True))
        # medición
        acc_meas.append(_accuracy_single_run(mult, env, proc=False))
    # Figuras 2 & 3
    for acc, name in zip([acc_proc, acc_meas],
                         ["Process noise", "Measurement noise"]):
        plt.figure(figsize=(4,3))
        plt.plot(noise_scales, acc, '-o')
        plt.ylim(0,1.05)
        plt.xlabel(f"{name} multiplier")
        plt.ylabel("Classifier accuracy")
        plt.title(f"Accuracy vs {name.lower()}")
        plt.tight_layout()
        plt.savefig(f"fig_{name.split()[0].lower()}.png", dpi=200)
    return acc_proc, acc_meas

def _accuracy_single_run(mult, env, proc=True):
    inertias = env.inertias
    # simulate each inertia once
    preds, truth = [], []
    for k,I in enumerate(inertias):
        omega = np.zeros(3)
        act_seq = np.random.choice(PWM_LEVELS, (SEQ_LEN, NUM_THRUST))
        traj = []
        for u in act_seq:
            torque = env._apply_torque(u, I)
            if proc:
                omega = rk4_step(omega, torque, I, DT) + \
                        np.random.normal(0, mult*1e-3, size=3)
            else:
                omega = rk4_step(omega, torque, I, DT)
            y = omega + (0 if proc else np.random.normal(0, mult*1e-3, size=3))
            traj.append(y)
        preds.append(traj)
        truth.append(k)
    model = TimeSeriesKMeans(n_clusters=N_MODELS,
                             metric="softdtw", max_iter=50,
                             random_state=0)
    y_pred = model.fit_predict(np.array(preds))
    return f1_score(truth, y_pred, average='micro')

# ---------- 6. Main ----------
if __name__ == "__main__":
    model, rewards = train_ppo()
    acc_proc, acc_meas = evaluate_accuracy(model, NOISE_LEVELS)
    print("Accuracy with 1× noise  (proc, meas):",
          acc_proc[NOISE_LEVELS.tolist().index(1.0)],
          acc_meas[NOISE_LEVELS.tolist().index(1.0)])
    plt.show()

