"""
SEMANTIC COMMUNICATION FOR CARTPOLE CONTROL
============================================
Complete Version - Agent A, Agent B, and Periodic Baselines

Features:
- Maximum GPU utilization through large batches and rollouts
- Agent A (Always Transmit) - Baseline with full information
- Agent B (Expiration Predictor) - Our semantic method
- Periodic agents - Fixed interval baselines
- Comprehensive comparison framework

Agents:
1. Agent A (Always Transmit) - Baseline
2. Agent B (Expiration Predictor) - Our Semantic Method  
3. Periodic (Fixed Interval) - Baseline
"""

import os
import time
import json
import math
import random
import hashlib
import numpy as np
from typing import Optional, Tuple, List, Dict, Callable
from collections import defaultdict, deque
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import autocast, GradScaler

try:
    import gymnasium as gym
    from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
except ImportError:
    import gym
    from gym.vector import SyncVectorEnv, AsyncVectorEnv

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from tqdm import tqdm


# =================================================================
# CONFIGURATION - OPTIMIZED FOR MAXIMUM GPU UTILIZATION
# =================================================================

CONFIG = {
    # ============================================================
    # HARDWARE CONFIGURATION
    # ============================================================
    "GPU_ID": 0,
    "SEED": 42,
    "USE_MIXED_PRECISION": True,
    "PIN_MEMORY": True,
    "NUM_WORKERS": 4,
    "PREFETCH_FACTOR": 2,
    "USE_TORCH_COMPILE": False,

    # ============================================================
    # PARALLELIZATION - KEY FOR GPU UTILIZATION
    # ============================================================
    "NUM_PARALLEL_ENVS": 50,
    "USE_ASYNC_ENVS": False,
    "ROLLOUT_LENGTH": 128,

    # ============================================================
    # AGENT CONTROL I call it Agent A
    # ============================================================
    "AGENT_A_MODE": "TRAIN",  # "TRAIN", "LOAD", "SKIP"
    "AGENT_A_PATH": "",
    

    # ============================================================
    # ENVIRONMENT PARAMETERS
    # ============================================================
    "MAX_STEPS_PER_EPISODE": 500,
    "OBSERVATION_NOISE_STD": 1,
    "TERMINATION_PENALTY": 50.0,       # For termination we get a reward of -50  

    # Target tracking (random walk)
    "TARGET_SIGMA": 0.02,
    "TARGET_KAPPA": 0.03,

    # ============================================================
    # TIME-VARYING NOISE
    # ============================================================
    "USE_TIME_VARYING_NOISE": True,
    "NOISE_VARIANCE_MAX": 0.3,
    "NUM_NOISE_LEVELS": 10,

    # ============================================================
    # EXPIRATION PARAMETERS
    # ============================================================
    "MAX_AGE": 5,
    "EXPIRATION_ROLLOUT_LENGTH": 8,
    "EXPIRATION_NUM_RUNS": 5,

    # Epsilon values for Agent B
    "EPSILON_VALUES": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],

    # Communication cost sweep
    "COMM_COST_VALUES": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],

    # ============================================================
    # TRAINING PARAMETERS - OPTIMIZED FOR GPU
    # ============================================================
    "ORACLE_EPISODES": 3000,
    "ORACLE_REWARD_THRESHOLD": 1000,
    "ORACLE_PATIENCE": 1000000,

    "EXPIRATION_DATASET_SIZE": 5000,
    "USE_DATASET_CACHE": True,

    "EVAL_EPISODES": 300,

    # ============================================================
    # NEURAL NETWORK ARCHITECTURE
    # ============================================================
    "HIDDEN": 256,
    "DEPTH": 5,

    # ============================================================
    # PPO HYPERPARAMETERS - OPTIMIZED FOR GPU UTILIZATION
    # ============================================================
    "PPO_GAMMA": 0.99,
    "PPO_LAMBDA": 0.95,
    "PPO_CLIP": 0.2,
    "PPO_LR": 3e-4,
    "PPO_TRAIN_EPOCHS": 10,
    "PPO_MINIBATCH": 8192,
    "ENTROPY_COEF": 0.01,
    "VALUE_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,

    # ============================================================
    # LEARNING RATE SCHEDULING
    # ============================================================
    "USE_LR_SCHEDULING": True,
    "LR_SCHEDULE_END_FACTOR": 0.1,

    # ============================================================
    # EXPIRATION PREDICTOR
    # ============================================================
    "PREDICTOR_INPUT_DIM": 13,
    "EXPIRATION_PREDICTOR_TRAIN_EPOCHS": 5000,
    "EXPIRATION_PREDICTOR_BATCH_SIZE": 1024,
    "EXPIRATION_PREDICTOR_LR": 3e-4,
    "EXPIRATION_PREDICTOR_PATIENCE": 30,

    # ============================================================
    # LOGGING
    # ============================================================
    "PRINT_EVERY_EP": 50,
    "PERFORMANCE_WINDOW": 100,

    # ============================================================
    # PERIODIC AGENTS
    # ============================================================
    "ENABLE_PERIODIC_AGENTS": True,
    "PERIODIC_INTERVALS": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}


# =================================================================
# UTILITIES
# =================================================================

def set_seed(seed: int):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def timestamp():
    """Get current timestamp string"""
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def unique_results_dir(base_name="semantic_comm"):
    """Create unique results directory"""
    user_home = os.path.expanduser("~")
    desktop = os.path.join(user_home, "Desktop")
    tag = f"{base_name}_{timestamp()}"
    
    if os.path.isdir(desktop) and os.access(desktop, os.W_OK):
        out_dir = os.path.join(desktop, tag)
    else:
        out_dir = os.path.join(os.getcwd(), tag)
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "cache"), exist_ok=True)
    
    return out_dir


def get_config_hash(cfg, keys=None):
    """Get hash of config for caching"""
    if keys is None:
        keys = ["MAX_AGE", "EXPIRATION_DATASET_SIZE", "NOISE_VARIANCE_MAX",
                "EXPIRATION_ROLLOUT_LENGTH", "EXPIRATION_NUM_RUNS"]
    
    config_str = "_".join(f"{k}={cfg.get(k, '')}" for k in sorted(keys))
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def print_gpu_stats():
    """Print GPU memory statistics"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            total = props.total_memory / 1e9
            print(f"GPU {i}: {props.name}")
            print(f"  Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")


class Logger:
    """Simple logger with file and console output"""
    
    def __init__(self, log_file):
        self.log_file = log_file
        self.start_time = time.time()
    
    def log(self, message, print_console=True):
        elapsed = time.time() - self.start_time
        timestamp_str = time.strftime("%H:%M:%S")
        full_message = f"[{timestamp_str}] {message}"
        
        if print_console:
            print(full_message, flush=True)
        
        with open(self.log_file, 'a') as f:
            f.write(full_message + '\n')
    
    def section(self, title):
        separator = "=" * 70
        self.log(f"\n{separator}")
        self.log(title)
        self.log(separator)


# =================================================================
# NOISE AND OBSERVATION UTILITIES
# =================================================================

def add_observation_noise(obs, cfg, noise_std=None):
    """Add observation noise with discrete levels."""
    if not cfg["USE_TIME_VARYING_NOISE"]:
        std = cfg["OBSERVATION_NOISE_STD"]
        if std > 0:
            noise = np.random.normal(0, std, size=obs.shape)
            return obs + noise, std
        return obs.copy(), 0.0
    
    if noise_std is None:
        num_levels = cfg["NUM_NOISE_LEVELS"]
        max_variance = cfg["NOISE_VARIANCE_MAX"]
        discrete_variances = np.linspace(0, max_variance, num_levels)
        variance = np.random.choice(discrete_variances)
        noise_std = np.sqrt(variance)
    
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, size=obs.shape)
        return obs + noise, noise_std
    
    return obs.copy(), 0.0


def add_observation_noise_batch(obs_batch, cfg):
    """Add observation noise to a batch of observations."""
    n = len(obs_batch)
    
    if not cfg["USE_TIME_VARYING_NOISE"]:
        std = cfg["OBSERVATION_NOISE_STD"]
        if std > 0:
            noise = np.random.normal(0, std, size=obs_batch.shape)
            return obs_batch + noise, np.full(n, std)
        return obs_batch.copy(), np.zeros(n)
    
    num_levels = cfg["NUM_NOISE_LEVELS"]
    max_variance = cfg["NOISE_VARIANCE_MAX"]
    discrete_variances = np.linspace(0, max_variance, num_levels)
    
    variances = np.random.choice(discrete_variances, size=n)
    noise_stds = np.sqrt(variances)
    
    noise = np.random.randn(*obs_batch.shape) * noise_stds[:, np.newaxis]
    
    return obs_batch + noise, noise_stds


def save_env_state(env):
    """Save environment state for rollback"""
    try:
        state = env.unwrapped.state
        if isinstance(state, tuple):
            return tuple(np.array(s).copy() if hasattr(s, 'copy') else s for s in state)
        return np.array(state).copy()
    except:
        return None


def restore_env_state(env, state):
    """Restore environment state"""
    try:
        if state is not None:
            env.unwrapped.state = state
    except:
        pass


# =================================================================
# FEATURE COMPUTATION FUNCTIONS
# =================================================================

def compute_predictor_features(obs, target, noise_std, last_action=0):
    """
    Compute 13 features for expiration prediction.
    
    Features:
        0-3: Raw observation (cart_pos, cart_vel, pole_angle, pole_vel)
        4: Target position
        5: Tracking error (target - cart_pos)
        6: Absolute cart velocity
        7: Absolute angular velocity  
        8: Absolute pole angle
        9: Instability indicator (|angle * angular_vel|)
        10: Edge proximity (|cart_pos| / 2.4)
        11: Noise standard deviation
        12: Last action
    """
    cart_pos, cart_vel, pole_angle, pole_vel = obs
    
    features = np.array([
        cart_pos,
        cart_vel,
        pole_angle,
        pole_vel,
        target,
        target - cart_pos,
        abs(cart_vel),
        abs(pole_vel),
        abs(pole_angle),
        abs(pole_angle * pole_vel),
        abs(cart_pos) / 2.4,
        noise_std,
        float(last_action),
    ], dtype=np.float32)
    
    return features


def compute_controller_features(obs, age, target, last_action, noise_std, max_age):
    """
    Compute 9 features for oracle controller.
    
    Features:
        0-3: Observation (possibly stale)
        4: Age (clamped to max_age)
        5: Target position
        6: Tracking error
        7: Last action
        8: Noise std
    """
    clamped_age = min(age, max_age)
    
    features = np.array([
        obs[0], obs[1], obs[2], obs[3],
        clamped_age,
        target,
        target - obs[0],
        float(last_action),
        noise_std,
    ], dtype=np.float32)
    
    return features


def compute_controller_features_batch(obs_batch, ages, targets, last_actions, noise_stds, max_age):
    """Compute controller features for a batch of observations."""
    n = len(obs_batch)
    features = np.zeros((n, 9), dtype=np.float32)
    
    clamped_ages = np.minimum(ages, max_age)
    
    features[:, 0:4] = obs_batch
    features[:, 4] = clamped_ages
    features[:, 5] = targets
    features[:, 6] = targets - obs_batch[:, 0]
    features[:, 7] = last_actions.astype(np.float32)
    features[:, 8] = noise_stds
    
    return features


# =================================================================
# CORE CLASSES
# =================================================================

class RandomWalkTarget:
    """Random walk target for cart position tracking"""
    
    def __init__(self, sigma, kappa, clamp=(-2.4, 2.4)):
        self.sigma = sigma
        self.kappa = kappa
        self.clamp = clamp
        self.T = 0.0
    
    def reset(self, x0):
        self.T = float(x0)
        return self.T
    
    def step(self, x_cart):
        self.T += np.random.normal(0.0, self.sigma) + self.kappa * (x_cart - self.T)
        self.T = float(np.clip(self.T, self.clamp[0], self.clamp[1]))
        return self.T


def mlp(sizes, act=nn.ReLU, out_act=nn.Identity):
    """Create MLP network"""
    layers = []
    for j in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[j], sizes[j + 1]))
        if j < len(sizes) - 2:
            layers.append(act())
        else:
            layers.append(out_act())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """
    Oracle policy network.
    Input: 9D (obs[4], age, target, error, last_action, noise_std)
    Output: Action logits (2D), Value (1D)
    """
    
    def __init__(self, obs_dim, n_actions, hidden, depth):
        super().__init__()
        
        self.body = mlp([obs_dim] + [hidden] * depth, nn.ReLU, nn.ReLU)
        
        self.pi = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )
        
        self.v = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)
    
    @torch.no_grad()
    def get_value(self, x):
        _, v = self.forward(x)
        return v
    
    @torch.no_grad()
    def act_deterministic(self, x):
        logits, _ = self.forward(x)
        return torch.argmax(logits, dim=-1)
    
    @torch.no_grad()
    def act_stochastic(self, x):
        logits, v = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), v


class ExpirationPredictor(nn.Module):
    """
    Predicts expiration time τ ∈ {1, 2, ..., max_age+1}
    
    Input: 13D predictor features
    Output: Class logits (num_classes = max_age + 1), Regression value
    """
    
    def __init__(self, in_dim, max_age, hidden, depth):
        super().__init__()
        self.max_age = max_age
        self.num_classes = max_age + 1
        
        layers = []
        prev_dim = in_dim
        for i in range(depth):
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(nn.ReLU())
            if i < depth - 1:
                layers.append(nn.Dropout(0.1))
            prev_dim = hidden
        self.body = nn.Sequential(*layers)
        
        self.class_head = nn.Linear(hidden, self.num_classes)
        
        self.reg_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )
    
    def forward(self, x):
        h = self.body(x)
        logits = self.class_head(h)
        reg_value = self.reg_head(h).squeeze(-1)
        return logits, reg_value
    
    @torch.no_grad()
    def predict_expiration(self, x, conservative=True):
        """Predict expiration time τ."""
        logits, reg_value = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        
        confidence, pred_class = probs.max(dim=-1)
        tau_argmax = pred_class + 1
        
        if conservative:
            class_values = torch.arange(1, self.num_classes + 1, device=x.device).float()
            tau_expected = (probs * class_values).sum(dim=-1)
            tau = tau_argmax.float() * confidence + tau_expected * (1 - confidence)
            tau = tau.round().clamp(1, self.max_age + 1).long()
        else:
            tau = tau_argmax
        
        return tau, confidence


# =================================================================
# MODEL UTILITIES
# =================================================================

def maybe_compile_model(model, cfg, mode="reduce-overhead"):
    """Optionally compile model with torch.compile for speedup."""
    if not cfg.get("USE_TORCH_COMPILE", False):
        return model
    
    if not hasattr(torch, 'compile'):
        return model
    
    if torch.cuda.is_available():
        device_cap = torch.cuda.get_device_capability()
        if device_cap[0] < 7:
            print(f"⚠️ GPU CUDA capability {device_cap[0]}.{device_cap[1]} < 7.0, skipping torch.compile")
            return model
    
    try:
        compiled = torch.compile(model, mode=mode)
        print(f"✅ Model compiled with mode='{mode}'")
        return compiled
    except Exception as e:
        print(f"⚠️ torch.compile failed: {e}")
        return model


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =================================================================
# ENVIRONMENT UTILITIES
# =================================================================

def build_env(seed):
    """Create single environment"""
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    return env


def make_env_fn(seed, rank):
    """Factory function for vectorized environments"""
    def _init():
        env = gym.make("CartPole-v1")
        env.reset(seed=seed + rank)
        return env
    return _init


def create_vec_envs(num_envs, seed, use_async=False):
    """Create vectorized environments"""
    env_fns = [make_env_fn(seed, i) for i in range(num_envs)]
    
    if use_async:
        try:
            envs = AsyncVectorEnv(env_fns)
            return envs, True
        except Exception as e:
            print(f"⚠️ AsyncVectorEnv failed: {e}, using SyncVectorEnv")
    
    return SyncVectorEnv(env_fns), False


# =================================================================
# PPO TRAINER - OPTIMIZED FOR GPU UTILIZATION
# =================================================================

class PPO:
    """Optimized PPO trainer with high GPU utilization."""
    
    def __init__(self, model, cfg, device, total_updates=None):
        self.model = model
        self.device = device
        self.cfg = cfg
        
        self.gamma = cfg["PPO_GAMMA"]
        self.lam = cfg["PPO_LAMBDA"]
        self.clip = cfg["PPO_CLIP"]
        self.entropy_coef = cfg["ENTROPY_COEF"]
        self.value_coef = cfg["VALUE_COEF"]
        self.max_grad_norm = cfg["MAX_GRAD_NORM"]
        self.epochs = cfg["PPO_TRAIN_EPOCHS"]
        self.mb_size = cfg["PPO_MINIBATCH"]
        
        self.optimizer = optim.AdamW(model.parameters(), lr=cfg["PPO_LR"], eps=1e-5)
        
        self.use_amp = cfg["USE_MIXED_PRECISION"] and device.type == 'cuda'
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        if cfg["USE_LR_SCHEDULING"] and total_updates:
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=cfg["LR_SCHEDULE_END_FACTOR"],
                total_iters=total_updates
            )
        else:
            self.scheduler = None
    
    @staticmethod
    def compute_gae_vectorized(rewards, values, dones, gamma, lam):
        """Compute GAE in a vectorized manner."""
        T, N = rewards.shape
        advantages = torch.zeros_like(rewards)
        lastgaelam = torch.zeros(N, device=rewards.device)
        
        for t in reversed(range(T)):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
            advantages[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        
        returns = advantages + values[:T]
        return advantages, returns
    
    @torch.no_grad()
    def act(self, obs_batch):
        """Sample actions for a batch of observations"""
        if self.use_amp:
            with autocast('cuda'):
                logits, v = self.model(obs_batch)
        else:
            logits, v = self.model(obs_batch)
        
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), v
    
    def update(self, obs, acts, logps_old, advantages, returns):
        """Update policy and value networks."""
        if len(obs) == 0:
            return {}
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        batch_size = len(obs)
        
        total_loss = 0
        total_pi_loss = 0
        total_v_loss = 0
        total_entropy = 0
        n_updates = 0
        
        for epoch in range(self.epochs):
            indices = torch.randperm(batch_size, device=self.device)
            
            for start in range(0, batch_size, self.mb_size):
                end = min(start + self.mb_size, batch_size)
                mb_idx = indices[start:end]
                
                self.optimizer.zero_grad()
                
                if self.use_amp:
                    with autocast('cuda'):
                        logits, v = self.model(obs[mb_idx])
                        dist = torch.distributions.Categorical(logits=logits)
                        
                        logp = dist.log_prob(acts[mb_idx])
                        entropy = dist.entropy().mean()
                        
                        ratio = torch.exp(logp - logps_old[mb_idx])
                        
                        surr1 = ratio * advantages[mb_idx]
                        surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages[mb_idx]
                        pi_loss = -torch.min(surr1, surr2).mean()
                        
                        v_loss = (returns[mb_idx] - v).pow(2).mean()
                        
                        loss = pi_loss + self.value_coef * v_loss - self.entropy_coef * entropy
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    logits, v = self.model(obs[mb_idx])
                    dist = torch.distributions.Categorical(logits=logits)
                    
                    logp = dist.log_prob(acts[mb_idx])
                    entropy = dist.entropy().mean()
                    
                    ratio = torch.exp(logp - logps_old[mb_idx])
                    
                    surr1 = ratio * advantages[mb_idx]
                    surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages[mb_idx]
                    pi_loss = -torch.min(surr1, surr2).mean()
                    
                    v_loss = (returns[mb_idx] - v).pow(2).mean()
                    
                    loss = pi_loss + self.value_coef * v_loss - self.entropy_coef * entropy
                    
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                total_loss += loss.item()
                total_pi_loss += pi_loss.item()
                total_v_loss += v_loss.item()
                total_entropy += entropy.item()
                n_updates += 1
        
        if self.scheduler:
            self.scheduler.step()
        
        return {
            'loss': total_loss / max(n_updates, 1),
            'pi_loss': total_pi_loss / max(n_updates, 1),
            'v_loss': total_v_loss / max(n_updates, 1),
            'entropy': total_entropy / max(n_updates, 1),
            'lr': self.optimizer.param_groups[0]['lr']
        }


# =================================================================
# EXPIRATION TIME MEASUREMENT
# =================================================================

def measure_expiration_time(env, oracle, obs, target, last_action, noise_std,
                            epsilon, max_age, horizon, num_runs, device, cfg):
    """Measure expiration time for an observation."""
    saved_state = save_env_state(env)
    expiration = 1
    
    for test_age in range(1, max_age + 1):
        degradations = []
        
        for run in range(num_runs):
            run_seed = np.random.randint(0, 1_000_000_000)
            
            # Phase 1: Age the sample
            np.random.seed(run_seed)
            restore_env_state(env, saved_state)
            
            target_proc = RandomWalkTarget(sigma=cfg["TARGET_SIGMA"], kappa=cfg["TARGET_KAPPA"])
            target_proc.T = target
            
            current_obs = obs.copy()
            current_action = last_action
            aging_failed = False
            
            for age_step in range(test_age):
                clamped_age = min(age_step, max_age)
                features = compute_controller_features(
                    current_obs, clamped_age, target_proc.T, current_action, noise_std, max_age
                )
                
                with torch.no_grad():
                    action = oracle.act_deterministic(
                        torch.as_tensor(features, device=device).unsqueeze(0)
                    )
                
                raw_next, _, terminated, _, _ = env.step(int(action.item()))
                
                if terminated:
                    aging_failed = True
                    break
                
                current_obs, _ = add_observation_noise(raw_next, cfg, noise_std)
                target_proc.step(float(current_obs[0]))
                current_action = int(action.item())
            
            if aging_failed:
                degradations.append(1000.0)
                continue
            
            aged_state = save_env_state(env)
            aged_obs = current_obs.copy()
            aged_target = target_proc.T
            aged_action = current_action
            rng_state = np.random.get_state()
            
            # Phase 2: V_fresh
            np.random.set_state(rng_state)
            restore_env_state(env, aged_state)
            
            target_proc_fresh = RandomWalkTarget(sigma=cfg["TARGET_SIGMA"], kappa=cfg["TARGET_KAPPA"])
            target_proc_fresh.T = aged_target
            
            V_fresh = 0.0
            obs_f = aged_obs.copy()
            action_f = aged_action
            
            for h in range(horizon):
                features = compute_controller_features(
                    obs_f, 0, target_proc_fresh.T, action_f, noise_std, max_age
                )
                
                with torch.no_grad():
                    action = oracle.act_deterministic(
                        torch.as_tensor(features, device=device).unsqueeze(0)
                    )
                
                raw_next, _, terminated, _, _ = env.step(int(action.item()))
                obs_f, _ = add_observation_noise(raw_next, cfg, noise_std)
                target_proc_fresh.step(float(obs_f[0]))
                
                reward = np.clip(1.0 - abs(obs_f[0] - target_proc_fresh.T), -1.0, 1.0)
                if terminated:
                    reward -= cfg["TERMINATION_PENALTY"]
                    V_fresh += reward
                    break
                
                V_fresh += reward
                action_f = int(action.item())
            
            # Phase 3: V_stale
            np.random.set_state(rng_state)
            restore_env_state(env, aged_state)
            
            target_proc_stale = RandomWalkTarget(sigma=cfg["TARGET_SIGMA"], kappa=cfg["TARGET_KAPPA"])
            target_proc_stale.T = aged_target
            
            V_stale = 0.0
            obs_s_true = aged_obs.copy()
            action_s = aged_action
            
            for h in range(horizon):
                clamped_age = min(test_age + h, max_age)
                features = compute_controller_features(
                    obs, clamped_age, target_proc_stale.T, action_s, noise_std, max_age
                )
                
                with torch.no_grad():
                    action = oracle.act_deterministic(
                        torch.as_tensor(features, device=device).unsqueeze(0)
                    )
                
                raw_next, _, terminated, _, _ = env.step(int(action.item()))
                obs_s_true, _ = raw_next
                target_proc_stale.step(float(obs_s_true[0]))
                
                reward = 1.0 - abs(obs_s_true[0] - target_proc_stale.T)
                if terminated:
                    reward -= cfg["TERMINATION_PENALTY"]
                    V_stale += reward
                    break
                
                V_stale += reward
                action_s = int(action.item())
            
            degradations.append(V_fresh - V_stale)
        
        avg_degradation = np.mean(degradations)
        
        if avg_degradation <= epsilon:
            expiration = test_age + 1
        else:
            break
    
    restore_env_state(env, saved_state)
    return min(expiration, max_age + 1)


# =================================================================
# ORACLE TRAINING - HIGH GPU UTILIZATION
# =================================================================

def train_oracle(results_dir, device, cfg, logger):
    """Train staleness-aware oracle policy using PPO."""
    logger.section("ORACLE TRAINING (Agent A)")
    
    num_envs = cfg["NUM_PARALLEL_ENVS"]
    max_age = cfg["MAX_AGE"]
    rollout_length = cfg.get("ROLLOUT_LENGTH", 128)
    
    samples_per_update = num_envs * rollout_length
    total_updates = cfg["ORACLE_EPISODES"]
    
    logger.log(f"Parallel Envs: {num_envs}")
    logger.log(f"Rollout Length: {rollout_length}")
    logger.log(f"Samples per Update: {samples_per_update:,}")
    logger.log(f"PPO Minibatch: {cfg['PPO_MINIBATCH']}")
    logger.log(f"PPO Epochs: {cfg['PPO_TRAIN_EPOCHS']}")
    logger.log(f"Max Age: {max_age}")
    
    oracle = ActorCritic(9, 2, cfg["HIDDEN"], cfg["DEPTH"]).to(device)
    oracle = maybe_compile_model(oracle, cfg)
    logger.log(f"Model Parameters: {count_parameters(oracle):,}")
    
    ppo = PPO(oracle, cfg, device, total_updates=total_updates)
    
    envs, is_async = create_vec_envs(num_envs, cfg["SEED"], cfg.get("USE_ASYNC_ENVS", False))
    logger.log(f"Environment type: {'Async' if is_async else 'Sync'}")
    
    target_procs = [RandomWalkTarget(sigma=cfg["TARGET_SIGMA"], kappa=cfg["TARGET_KAPPA"])
                    for _ in range(num_envs)]
    
    # Pre-allocate GPU buffers
    obs_buffer = torch.zeros((rollout_length, num_envs, 9), device=device, dtype=torch.float32)
    act_buffer = torch.zeros((rollout_length, num_envs), device=device, dtype=torch.long)
    logp_buffer = torch.zeros((rollout_length, num_envs), device=device, dtype=torch.float32)
    rew_buffer = torch.zeros((rollout_length, num_envs), device=device, dtype=torch.float32)
    val_buffer = torch.zeros((rollout_length + 1, num_envs), device=device, dtype=torch.float32)
    done_buffer = torch.zeros((rollout_length, num_envs), device=device, dtype=torch.float32)
    
    obs_histories = [deque(maxlen=max_age + 2) for _ in range(num_envs)]
    noise_histories = [deque(maxlen=max_age + 2) for _ in range(num_envs)]
    
    recent_rewards = deque(maxlen=cfg["PERFORMANCE_WINDOW"])
    best_avg_reward = -float('inf')
    best_model_state = None
    patience_counter = 0
    
    episode_rewards = np.zeros(num_envs)
    episode_lengths = np.zeros(num_envs, dtype=np.int32)
    
    start_time = time.time()
    total_steps = 0
    
    # Initial reset
    reset_result = envs.reset()
    raw_obs_batch = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    
    local_obs, noise_stds = add_observation_noise_batch(raw_obs_batch, cfg)
    targets = np.array([target_procs[i].reset(float(local_obs[i][0])) for i in range(num_envs)])
    last_actions = np.zeros(num_envs, dtype=np.int32)
    
    for i in range(num_envs):
        obs_histories[i].append(local_obs[i].copy())
        noise_histories[i].append(noise_stds[i])
    
    pbar = tqdm(range(total_updates), desc="Oracle Training", ncols=120)
    
    for update in pbar:
        # Age curriculum
        progress = update / total_updates
        if progress < 0.25:
            curr_max_age = 0
        elif progress < 0.5:
            curr_max_age = 1
        elif progress < 0.75:
            curr_max_age = min(2, max_age)
        else:
            curr_max_age = max_age
        
        # Rollout phase
        for step in range(rollout_length):
            features_np = np.zeros((num_envs, 9), dtype=np.float32)
            
            for i in range(num_envs):
                age = np.random.randint(0, min(curr_max_age + 1, len(obs_histories[i])))
                stale_obs = obs_histories[i][-(age + 1)]
                stale_noise = noise_histories[i][-(age + 1)]
                
                features_np[i] = compute_controller_features(
                    stale_obs, age, targets[i], last_actions[i], stale_noise, max_age
                )
            
            features_tensor = torch.as_tensor(features_np, device=device)
            
            with torch.no_grad():
                if ppo.use_amp:
                    with autocast('cuda'):
                        logits, values = oracle(features_tensor)
                else:
                    logits, values = oracle(features_tensor)
                
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
            
            obs_buffer[step] = features_tensor
            val_buffer[step] = values
            act_buffer[step] = actions
            logp_buffer[step] = log_probs
            
            actions_np = actions.cpu().numpy()
            step_result = envs.step(actions_np)
            raw_next_obs = step_result[0]
            terminateds = step_result[2]
            truncateds = step_result[3]
            
            next_local_obs, next_noise_stds = add_observation_noise_batch(raw_next_obs, cfg)
            
            for i in range(num_envs):
                targets[i] = target_procs[i].step(float(next_local_obs[i][0]))
                
                tracking_error = abs(next_local_obs[i][0] - targets[i])
                reward = np.clip(1.0 - tracking_error, -1.0, 1.0)
                
                if terminateds[i]:
                    reward -= cfg["TERMINATION_PENALTY"]
                
                rew_buffer[step, i] = reward
                done_buffer[step, i] = float(terminateds[i] or truncateds[i])
                
                episode_rewards[i] += reward
                episode_lengths[i] += 1
                
                if terminateds[i] or truncateds[i]:
                    recent_rewards.append(episode_rewards[i])
                    episode_rewards[i] = 0
                    episode_lengths[i] = 0
                    
                    obs_histories[i].clear()
                    noise_histories[i].clear()
                    targets[i] = target_procs[i].reset(float(next_local_obs[i][0]))
                
                obs_histories[i].append(next_local_obs[i].copy())
                noise_histories[i].append(next_noise_stds[i])
                last_actions[i] = actions_np[i]
            
            local_obs = next_local_obs
            noise_stds = next_noise_stds
            total_steps += num_envs
        
        # Bootstrap value
        with torch.no_grad():
            final_features_np = compute_controller_features_batch(
                local_obs, np.zeros(num_envs), targets, last_actions, noise_stds, max_age
            )
            final_features = torch.as_tensor(final_features_np, device=device)
            
            if ppo.use_amp:
                with autocast('cuda'):
                    _, final_values = oracle(final_features)
            else:
                _, final_values = oracle(final_features)
            
            val_buffer[rollout_length] = final_values
        
        # Compute GAE
        advantages, returns = PPO.compute_gae_vectorized(
            rew_buffer, val_buffer, done_buffer, ppo.gamma, ppo.lam
        )
        
        # Flatten for training
        b_obs = obs_buffer.reshape(-1, 9)
        b_actions = act_buffer.reshape(-1)
        b_logprobs = logp_buffer.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        
        # PPO update
        metrics = ppo.update(b_obs, b_actions, b_logprobs, b_advantages, b_returns)
        
        # Logging
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0
        
        pbar.set_postfix({
            'reward': f'{avg_reward:.1f}',
            'age': curr_max_age,
            'loss': f'{metrics.get("loss", 0):.3f}',
            'steps': f'{total_steps/1e6:.2f}M'
        })
        
        if (update + 1) % cfg["PRINT_EVERY_EP"] == 0:
            elapsed = (time.time() - start_time) / 60
            steps_per_sec = total_steps / (time.time() - start_time)
            
            logger.log(
                f"Update {update+1:4d} | Reward: {avg_reward:6.1f} | "
                f"Age: {curr_max_age} | Steps: {total_steps/1e6:.2f}M | "
                f"SPS: {steps_per_sec:,.0f} | Time: {elapsed:.1f}m"
            )
        
        # Save best model
        if len(recent_rewards) >= 50 and avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            best_model_state = {k: v.cpu().clone() for k, v in oracle.state_dict().items()}
            patience_counter = 0
        elif len(recent_rewards) >= 50:
            patience_counter += 1
        
        # Early stopping
        if (patience_counter >= cfg["ORACLE_PATIENCE"] and 
            avg_reward >= cfg["ORACLE_REWARD_THRESHOLD"]):
            logger.log(f"\n✓ Early stopping at update {update+1}")
            logger.log(f"  Best reward: {best_avg_reward:.1f}")
            break
    
    pbar.close()
    envs.close()
    
    if best_model_state is not None:
        oracle.load_state_dict(best_model_state)
    
    oracle_path = os.path.join(results_dir, "unified_oracle.pt")
    torch.save(oracle.state_dict(), oracle_path)
    
    elapsed = (time.time() - start_time) / 60
    logger.log(f"\n✓ Oracle training complete ({elapsed:.1f} min)")
    logger.log(f"  Total steps: {total_steps/1e6:.2f}M")
    logger.log(f"  Best reward: {best_avg_reward:.1f}")
    logger.log(f"  Saved to: {oracle_path}")
    
    return oracle


# =================================================================
# DATASET GENERATION
# =================================================================

def get_dataset_cache_path(results_dir, epsilon, cfg):
    """Get cache path for dataset"""
    cache_dir = os.path.join(results_dir, "cache")
    config_hash = get_config_hash(cfg)
    return os.path.join(cache_dir, f"dataset_eps{epsilon:.3f}_{config_hash}.pt")


def load_cached_dataset(results_dir, epsilon, cfg):
    """Load dataset from cache if available"""
    if not cfg.get("USE_DATASET_CACHE", True):
        return None
    
    cache_path = get_dataset_cache_path(results_dir, epsilon, cfg)
    if os.path.exists(cache_path):
        try:
            data = torch.load(cache_path, weights_only=False)
            return data
        except:
            return None
    return None


def save_dataset_cache(results_dir, epsilon, dataset, cfg):
    """Save dataset to cache"""
    if not cfg.get("USE_DATASET_CACHE", True):
        return
    
    cache_path = get_dataset_cache_path(results_dir, epsilon, cfg)
    torch.save(dataset, cache_path)


# =================================================================
# AGENT SESSIONS
# =================================================================

class AgentASession:
    """Agent A: Always transmits"""
    def __init__(self, env, cfg, device):
        self.env = env
        self.cfg = cfg
        self.device = device
        self.max_age = cfg["MAX_AGE"]
        self.target_proc = RandomWalkTarget(sigma=cfg["TARGET_SIGMA"], kappa=cfg["TARGET_KAPPA"])
        self.obs, self.noise, self.target, self.last_action = None, 0.0, 0.0, 0
    
    def reset(self, seed=None):
        raw, _ = self.env.reset(seed=seed)
        self.obs, self.noise = add_observation_noise(raw, self.cfg)
        self.target = self.target_proc.reset(float(self.obs[0]))
        self.last_action = 0
        return self._features()
    
    def _features(self):
        return compute_controller_features(self.obs, 0, self.target, self.last_action, self.noise, self.max_age)
    
    def step(self, action):
        self.last_action = int(action)
        raw, _, term, trunc, _ = self.env.step(action)
        self.obs, self.noise = add_observation_noise(raw, self.cfg)
        self.target = self.target_proc.step(float(self.obs[0]))
        reward = np.clip(1.0 - abs(self.obs[0] - self.target), -1.0, 1.0)
        if term: reward -= self.cfg["TERMINATION_PENALTY"]
        return self._features(), reward, term or trunc, {'transmitted': True}


# =================================================================
# EVALUATION
# =================================================================

def evaluate_agent(session_factory, oracle, cfg, device, episodes):
    """Generic evaluation loop"""
    oracle.eval()
    ctrl_rews, txs, steps = [], [], []
    
    for _ in range(episodes):
        session = session_factory()
        session.reset(seed=random.randint(0, 100000))
        ep_rew, ep_tx = 0, 1
        
        for i in range(cfg["MAX_STEPS_PER_EPISODE"]):
            if hasattr(session, 'maybe_transmit'): 
                session.maybe_transmit()
            
            feat = session._features()
            
            with torch.no_grad():
                a = oracle.act_deterministic(torch.as_tensor(feat, device=device).unsqueeze(0))
            
            res = session.step(int(a.item()))
            _, rew, done, info = res
            
            if info.get('transmitted', False):
                ep_tx += 1
            
            ep_rew += rew
            if done: 
                break
        
        ctrl_rews.append(ep_rew)
        txs.append(ep_tx)
        steps.append(i + 1)
    
    return {
        'control_reward': np.mean(ctrl_rews),
        'transmissions': np.mean(txs),
        'tx_rate': np.mean(txs) / np.mean(steps),
        'steps': np.mean(steps)
    }


# =================================================================
# PLOTTING
# =================================================================

def plot_results(all_results, results_dir, cfg, logger):
    """Generate summary plots"""
    if not all_results:
        return
    logger.section("PLOTTING")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extract data
    costs = sorted(set(r['comm_cost'] for r in all_results))
    
    # Plot 1: Net reward vs cost
    ax1 = axes[0]
    
    agent_a_net = []
    agent_b_net = []
    periodic_nets = {p: [] for p in cfg["PERIODIC_INTERVALS"]}
    
    for cost in costs:
        cost_results = [r for r in all_results if r['comm_cost'] == cost]
        if not cost_results:
            continue
        
        # Agent A
        r_a = cost_results[0]['agent_a']
        agent_a_net.append(r_a['control_reward'] - cost * r_a['transmissions'])
        
        # Agent B (best epsilon)
        best_b_net = -float('inf')
        for r in cost_results:
            if 'agent_b' in r:
                net = r['agent_b']['control_reward'] - cost * r['agent_b']['transmissions']
                if net > best_b_net:
                    best_b_net = net
        agent_b_net.append(best_b_net)
        
        # Periodic agents
        for period in cfg["PERIODIC_INTERVALS"]:
            key = f'periodic_{period}'
            if key in cost_results[0]:
                r_p = cost_results[0][key]
                periodic_nets[period].append(r_p['control_reward'] - cost * r_p['transmissions'])
    
    ax1.plot(costs[:len(agent_a_net)], agent_a_net, 'b-o', label='Agent A (Always)', linewidth=2, markersize=8)
    ax1.plot(costs[:len(agent_b_net)], agent_b_net, 'g-s', label='Agent B (Ours)', linewidth=2, markersize=8)
    
    for period in [2, 5, 10]:
        if period in periodic_nets and periodic_nets[period]:
            ax1.plot(costs[:len(periodic_nets[period])], periodic_nets[period], 
                    '--', label=f'Periodic-{period}', alpha=0.7)
    
    ax1.set_xlabel('Communication Cost', fontsize=12)
    ax1.set_ylabel('Net Reward', fontsize=12)
    ax1.set_title('Net Reward vs Communication Cost', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Transmission rate vs cost
    ax2 = axes[1]
    
    agent_a_tx = []
    agent_b_tx = []
    
    for cost in costs:
        cost_results = [r for r in all_results if r['comm_cost'] == cost]
        if not cost_results:
            continue
        
        agent_a_tx.append(cost_results[0]['agent_a']['tx_rate'])
        
        best_b_net = -float('inf')
        best_b_tx = 1.0
        for r in cost_results:
            if 'agent_b' in r:
                net = r['agent_b']['control_reward'] - cost * r['agent_b']['transmissions']
                if net > best_b_net:
                    best_b_net = net
                    best_b_tx = r['agent_b']['tx_rate']
        agent_b_tx.append(best_b_tx)
    
    ax2.plot(costs[:len(agent_a_tx)], agent_a_tx, 'b-o', label='Agent A', linewidth=2, markersize=8)
    ax2.plot(costs[:len(agent_b_tx)], agent_b_tx, 'g-s', label='Agent B', linewidth=2, markersize=8)
    
    ax2.set_xlabel('Communication Cost', fontsize=12)
    ax2.set_ylabel('Transmission Rate', fontsize=12)
    ax2.set_title('Transmission Rate vs Communication Cost', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "figures", "comparison.png"), dpi=150)
    plt.savefig(os.path.join(results_dir, "figures", "comparison.pdf"))
    logger.log("✓ Saved comparison.png and comparison.pdf")
    plt.close()


# =================================================================
# MAIN PIPELINE
# =================================================================

def main():
    gpu_id = CONFIG["GPU_ID"]
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    set_seed(CONFIG["SEED"])
    
    res_dir = unique_results_dir()
    logger = Logger(os.path.join(res_dir, "log.txt"))
    logger.section("STALENESS-AWARE CONTROLLER TRAINING")
    
    # 1. TRAIN ORACLE (Agent A)
    # This trains the controller to handle staleness via random age sampling
    if CONFIG["AGENT_A_MODE"] == "TRAIN":
        oracle = train_oracle(res_dir, device, CONFIG, logger)
        logger.log(f"✓ Training Complete. Model saved to {res_dir}")
    else:
        logger.log("Agent A mode not set to TRAIN. Exiting.")

if __name__ == "__main__":
    main()
