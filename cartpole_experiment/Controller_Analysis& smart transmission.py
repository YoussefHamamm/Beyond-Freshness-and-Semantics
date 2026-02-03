import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import json
from datetime import datetime
import time
from scipy import stats
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from tqdm import tqdm, trange
from tqdm.auto import tqdm as tqdm_auto
import sys

# =================================================================
# CONFIGURATION
# =================================================================
CONFIG = {
    "CONTROLLER_PATH": "",
    
    # Save/Load paths
    "SAVE_DIR": "",
    "PREDICTOR_SAVE_PATH": "",
    "DATASET_SAVE_PATH": "",
    "RESULTS_SAVE_PATH": "",
    "CONFIG_SAVE_PATH": "",
    
    # ═══════════════════════════════════════════════════════════════
    # CRASH BEHAVIOR CONFIGURATION
    # ═══════════════════════════════════════════════════════════════
    "END_ON_CRASH": True,
    "CRASH_TERMINAL_REWARD": -50.0,
    "CRASH_PENALTY": -10.0,
    
    # ═══════════════════════════════════════════════════════════════
    # Epsilon values
    "EPSILON_VALUES": np.linspace(0, 0.04, 101).tolist(),
#[0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011,   0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02,
                       #0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
    
    # Communication costs
    "COMMUNICATION_COSTS":[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] ,# [0.0, 0.2,  0.4,  0.6,  0.8,  1.0]
    
    # Periodic baselines
    "PERIODIC_INTERVALS": [1, 2, 3, 4, 5],
    
    # Data collection - VECTORIZED
    "NUM_TOTAL_SAMPLES": 1500000,
    "BATCH_SIZE_COLLECTION": 300,
    
    # MC simulation
    "NUM_MC_ROLLOUTS": 100,
    "ROLLOUT_HORIZON": 10,
    "MAX_EXPIRATION": 10,
    
    # Predictor training
    "PREDICTOR_HIDDEN": 128,
    "PREDICTOR_LAYERS": 5,
    "PREDICTOR_EPOCHS": 200,
    "PREDICTOR_LR": 0.0001,
    "PREDICTOR_BATCH_SIZE": 256,
    "PREDICTOR_WEIGHT_DECAY": 1e-3,
    "PREDICTOR_DROPOUT": 0.1,
    "EARLY_STOPPING_PATIENCE": 30,
    
    # Feature dimension
    "PREDICTOR_INPUT_DIM": 18,
    
    # Game settings
    "GAME_EPISODES": 1000,
    "MAX_EPISODE_STEPS": 500,
    
    # Noise settings
    "MAX_NOISE_STD": 0.5,
    "NUM_NOISE_LEVELS": 11,
    "NOISE_STD_VALUES": None,
    
    # System settings
    "MAX_AGE": 8,
    "HIDDEN": 256,
    "DEPTH": 3,
    "RNN_HIDDEN": 128,
    "MEMORY_LENGTH": 16,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    
    # Parallelization settings
    "NUM_PARALLEL_ENVS": 50,
    
    # Random seed
    "RANDOM_SEED": 42,
    
    # Statistics printing interval
    "STATS_PRINT_INTERVAL": 10,
}

CONFIG["NOISE_STD_VALUES"] = np.linspace(0, CONFIG["MAX_NOISE_STD"], CONFIG["NUM_NOISE_LEVELS"])
os.makedirs(CONFIG["SAVE_DIR"], exist_ok=True)


# =================================================================
# 2. REQUIRED CLASSES
# =================================================================
# =================================================================
# PRETTY PRINTING UTILITIES
# =================================================================

class PrettyPrinter:
    """Utility class for consistent, clean output formatting."""
    
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    @staticmethod
    def header(text, width=80):
        """Print a prominent header."""
        print("\n" + "═" * width)
        padding = (width - len(text) - 2) // 2
        print("║" + " " * padding + text + " " * (width - padding - len(text) - 2) + "║")
        print("═" * width)
    
    @staticmethod
    def subheader(text, width=80):
        """Print a subheader."""
        print("\n" + "─" * width)
        print(f"  {text}")
        print("─" * width)
    
    @staticmethod
    def section(text, char="─", width=60):
        """Print a section divider."""
        print(f"\n{char * 3} {text} {char * (width - len(text) - 5)}")
    
    @staticmethod
    def success(text):
        """Print success message."""
        print(f"  ✓ {text}")
    
    @staticmethod
    def info(text):
        """Print info message."""
        print(f"  ℹ {text}")
    
    @staticmethod
    def warning(text):
        """Print warning message."""
        print(f"  ⚠ {text}")
    
    @staticmethod
    def error(text):
        """Print error message."""
        print(f"  ✗ {text}")
    
    @staticmethod
    def keyvalue(key, value, key_width=30):
        """Print a key-value pair."""
        print(f"  {key:<{key_width}} : {value}")
    
    @staticmethod
    def table_header(columns, widths):
        """Print table header."""
        header = " │ ".join(f"{col:^{w}}" for col, w in zip(columns, widths))
        separator = "─┼─".join("─" * w for w in widths)
        print(f" {header}")
        print(f" {separator}")
    
    @staticmethod
    def table_row(values, widths, formats=None):
        """Print table row."""
        if formats is None:
            formats = [f"^{w}" for w in widths]
        cells = []
        for val, w, fmt in zip(values, widths, formats):
            if isinstance(val, float):
                cells.append(f"{val:{fmt}}")
            else:
                cells.append(f"{val:^{w}}")
        print(f" {' │ '.join(cells)}")
    
    @staticmethod
    def progress_box(title, stats_dict, width=60):
        """Print a stats box."""
        print(f"\n┌{'─' * (width-2)}┐")
        print(f"│ {title:<{width-4}} │")
        print(f"├{'─' * (width-2)}┤")
        for key, value in stats_dict.items():
            if isinstance(value, float):
                line = f"│  {key}: {value:.4f}"
            else:
                line = f"│  {key}: {value}"
            print(f"{line:<{width-1}}│")
        print(f"└{'─' * (width-2)}┘")
    
    @staticmethod
    def final_box(title, content_lines, width=70):
        """Print a final summary box."""
        print(f"\n╔{'═' * (width-2)}╗")
        print(f"║ {title:^{width-4}} ║")
        print(f"╠{'═' * (width-2)}╣")
        for line in content_lines:
            print(f"║ {line:<{width-4}} ║")
        print(f"╚{'═' * (width-2)}╝")


pp = PrettyPrinter()
def print_crash_behavior_info():
    """Print information about the current crash behavior setting."""
    pp.header("CRASH BEHAVIOR CONFIGURATION")
    
    if CONFIG["END_ON_CRASH"]:
        pp.keyvalue("Mode", "END ON CRASH")
        pp.keyvalue("Behavior", "Rollout/Episode terminates on crash")
        pp.keyvalue("Crash Reward", f"{CONFIG['CRASH_TERMINAL_REWARD']}")
    else:
        pp.keyvalue("Mode", "CONTINUE AFTER CRASH")
        pp.keyvalue("Behavior", "Rollout/Episode continues after crash")
        pp.keyvalue("Crash Penalty", f"{CONFIG['CRASH_PENALTY']}")
        
def print_expiration_statistics(epsilon_values, epsilon_stats, noise_stats, sample_count, compact=False):
    """Print detailed expiration statistics in a clean format."""
    if compact:
        # Compact version for progress updates
        all_exps = []
        for eps_idx in epsilon_stats:
            all_exps.extend(epsilon_stats[eps_idx])
        if all_exps:
            exp_arr = np.array(all_exps)
            return {
                'mean': exp_arr.mean(),
                'std': exp_arr.std(),
                'min': exp_arr.min(),
                'max': exp_arr.max()
            }
        return None
    
    pp.subheader(f"EXPIRATION STATISTICS ({sample_count:,} samples)")
    
    # Per-Epsilon Statistics Table
    print("\n  Per-Epsilon Statistics:")
    columns = ["ε", "Min", "Max", "Mean", "Std", "Median", "Q25", "Q75"]
    widths = [10, 6, 6, 8, 8, 8, 6, 6]
    pp.table_header(columns, widths)
    
    # Show subset of epsilons to avoid clutter
    eps_to_show = epsilon_values[::max(1, len(epsilon_values)//10)]  # Show ~10 epsilons
    if epsilon_values[-1] not in eps_to_show:
        eps_to_show = list(eps_to_show) + [epsilon_values[-1]]
    
    for eps in eps_to_show:
        eps_idx = epsilon_values.index(eps)
        exp_arr = np.array(epsilon_stats[eps_idx])
        if len(exp_arr) > 0:
            values = [
                f"{eps:.4f}",
                f"{exp_arr.min():.1f}",
                f"{exp_arr.max():.1f}",
                f"{exp_arr.mean():.2f}",
                f"{exp_arr.std():.2f}",
                f"{np.median(exp_arr):.2f}",
                f"{np.percentile(exp_arr, 25):.1f}",
                f"{np.percentile(exp_arr, 75):.1f}"
            ]
            print(f"  {' │ '.join(f'{v:^{w}}' for v, w in zip(values, widths))}")
    
    # Per-Noise Statistics
    print("\n  Per-Noise Level Statistics:")
    columns = ["Noise σ", "Count", "Mean τ*", "Std"]
    widths = [10, 10, 12, 10]
    pp.table_header(columns, widths)
    
    for noise_level in sorted(noise_stats.keys()):
        exp_data = noise_stats[noise_level]
        if len(exp_data) > 0:
            exp_arr = np.array(exp_data)
            values = [
                f"{noise_level:.3f}",
                f"{len(exp_arr):,}",
                f"{exp_arr.mean():.2f}",
                f"{exp_arr.std():.2f}"
            ]
            print(f"  {' │ '.join(f'{v:^{w}}' for v, w in zip(values, widths))}")        
class RandomWalkTarget:
    def __init__(self, sigma=0.02, kappa=0.03, clamp=(-2.4, 2.4)):
        self.sigma, self.kappa, self.clamp = sigma, kappa, clamp
        self.T = 0.0

    def reset(self, x0):
        self.T = float(x0)
        return self.T

    def step(self, x_cart):
        self.T += np.random.normal(0.0, self.sigma) + self.kappa * (x_cart - self.T)
        self.T = float(np.clip(self.T, self.clamp[0], self.clamp[1]))
        return self.T


class RecurrentActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden, depth, rnn_hidden):
        super().__init__()
        self.obs_dim = obs_dim
        self.rnn_hidden = rnn_hidden

        encoder_layers = []
        prev_dim = obs_dim
        for i in range(depth):
            encoder_layers.extend([nn.Linear(prev_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden)])
            prev_dim = hidden
        self.encoder = nn.Sequential(*encoder_layers)

        self.gru = nn.GRU(input_size=hidden, hidden_size=rnn_hidden, num_layers=1, batch_first=True)
        self.gru_ln = nn.LayerNorm(rnn_hidden)

        self.pi = nn.Sequential(nn.Linear(rnn_hidden, hidden), nn.ReLU(), nn.LayerNorm(hidden), nn.Linear(hidden, n_actions))
        self.v = nn.Sequential(nn.Linear(rnn_hidden, hidden), nn.ReLU(), nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x, hidden=None):
        batch_size, seq_len = x.shape[0], x.shape[1]
        x_flat = x.reshape(-1, self.obs_dim)
        encoded = self.encoder(x_flat).reshape(batch_size, seq_len, -1)
        if hidden is None:
            hidden = torch.zeros(1, batch_size, self.rnn_hidden, device=x.device)
        gru_out, new_hidden = self.gru(encoded, hidden)
        last_out = self.gru_ln(gru_out[:, -1, :])
        return self.pi(last_out), self.v(last_out).squeeze(-1), new_hidden

    @torch.no_grad()
    def act_deterministic(self, x, hidden=None):
        logits, _, new_hidden = self.forward(x, hidden)
        return torch.argmax(logits, dim=-1), new_hidden


class SimpleExpirationPredictor(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=128, num_layers=2,
                 num_epsilons=21, dropout=0.3, max_age=5):
        super().__init__()
        self.num_epsilons = num_epsilons
        self.max_age = max_age

        self.epsilon_embed = nn.Embedding(num_epsilons, 16)

        layers = []
        prev_dim = input_dim + 16
        for i in range(num_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        self.head = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, epsilon_idx=None):
        batch_size = x.shape[0]

        if epsilon_idx is not None:
            eps_embed = self.epsilon_embed(
                torch.full((batch_size,), epsilon_idx, dtype=torch.long, device=x.device)
            )
            h = torch.cat([x, eps_embed], dim=-1)
            h = self.backbone(h)
            pred = self.head(h).squeeze(-1)
            return torch.clamp(pred, 0, self.max_age)
        else:
            preds = []
            for eps_idx in range(self.num_epsilons):
                eps_embed = self.epsilon_embed(
                    torch.full((batch_size,), eps_idx, dtype=torch.long, device=x.device)
                )
                h = torch.cat([x, eps_embed], dim=-1)
                h = self.backbone(h)
                pred = self.head(h).squeeze(-1)
                preds.append(pred)
            return torch.clamp(torch.stack(preds, dim=1), 0, self.max_age)


# =================================================================
# 3. FEATURE ENGINEERING
# =================================================================
def compute_predictor_features_v3(noisy_obs, target, prev_obs=None, dt=0.02):
    """Enhanced features with physics-based noise discrimination."""
    features = np.zeros(18, dtype=np.float32)

    cart_pos, cart_vel, pole_angle, pole_ang_vel = noisy_obs

    features[0] = cart_pos
    features[1] = cart_vel
    features[2] = pole_angle
    features[3] = pole_ang_vel
    features[4] = target
    features[5] = target - cart_pos
    features[6] = abs(pole_angle)
    features[7] = abs(pole_ang_vel)
    features[8] = 0.5 * (cart_vel**2 + pole_ang_vel**2)
    features[9] = abs(cart_pos) / 2.4
    features[10] = abs(pole_angle) / 0.21
    features[11] = np.tanh(-pole_angle * pole_ang_vel * 5)

    extremeness = ((cart_pos / 0.8)**2 + (cart_vel / 1.0)**2 +
                   (pole_angle / 0.1)**2 + (pole_ang_vel / 1.0)**2)
    features[12] = np.tanh(extremeness / 4)

    pos_vel_sign = np.sign(cart_pos) * np.sign(cart_vel)
    features[13] = pos_vel_sign * min(abs(cart_pos), 1.0)

    angle_angvel_sign = np.sign(pole_angle) * np.sign(pole_ang_vel)
    features[14] = angle_angvel_sign * min(abs(pole_angle) * 10, 1.0)

    sensitivity = (abs(cart_pos)/2.4 + abs(pole_angle)/0.21) * \
                  (1 + abs(cart_vel) + abs(pole_ang_vel))
    features[15] = np.tanh(sensitivity)

    tolerance = 1.0 / (1.0 + abs(cart_pos) + abs(cart_vel) +
                       abs(pole_angle)*10 + abs(pole_ang_vel))
    features[16] = tolerance

    if prev_obs is not None:
        prev_pos, prev_vel, prev_angle, prev_ang_vel = prev_obs
        expected_pos = prev_pos + prev_vel * dt
        expected_angle = prev_angle + prev_ang_vel * dt
        physics_error = (abs(cart_pos - expected_pos) +
                         abs(pole_angle - expected_angle) * 10)
        features[17] = np.tanh(physics_error * 5)
    else:
        features[17] = 0.5

    return features


def compute_predictor_features_v3_batch(noisy_obs_batch, target_batch, prev_obs_batch=None, dt=0.02):
    """Vectorized feature computation for batch of observations."""
    batch_size = noisy_obs_batch.shape[0]
    features = np.zeros((batch_size, 18), dtype=np.float32)

    cart_pos = noisy_obs_batch[:, 0]
    cart_vel = noisy_obs_batch[:, 1]
    pole_angle = noisy_obs_batch[:, 2]
    pole_ang_vel = noisy_obs_batch[:, 3]

    features[:, 0] = cart_pos
    features[:, 1] = cart_vel
    features[:, 2] = pole_angle
    features[:, 3] = pole_ang_vel
    features[:, 4] = target_batch
    features[:, 5] = target_batch - cart_pos
    features[:, 6] = np.abs(pole_angle)
    features[:, 7] = np.abs(pole_ang_vel)
    features[:, 8] = 0.5 * (cart_vel**2 + pole_ang_vel**2)
    features[:, 9] = np.abs(cart_pos) / 2.4
    features[:, 10] = np.abs(pole_angle) / 0.21
    features[:, 11] = np.tanh(-pole_angle * pole_ang_vel * 5)

    extremeness = ((cart_pos / 0.8)**2 + (cart_vel / 1.0)**2 +
                   (pole_angle / 0.1)**2 + (pole_ang_vel / 1.0)**2)
    features[:, 12] = np.tanh(extremeness / 4)

    pos_vel_sign = np.sign(cart_pos) * np.sign(cart_vel)
    features[:, 13] = pos_vel_sign * np.minimum(np.abs(cart_pos), 1.0)

    angle_angvel_sign = np.sign(pole_angle) * np.sign(pole_ang_vel)
    features[:, 14] = angle_angvel_sign * np.minimum(np.abs(pole_angle) * 10, 1.0)

    sensitivity = (np.abs(cart_pos)/2.4 + np.abs(pole_angle)/0.21) * \
                  (1 + np.abs(cart_vel) + np.abs(pole_ang_vel))
    features[:, 15] = np.tanh(sensitivity)

    tolerance = 1.0 / (1.0 + np.abs(cart_pos) + np.abs(cart_vel) +
                       np.abs(pole_angle)*10 + np.abs(pole_ang_vel))
    features[:, 16] = tolerance

    if prev_obs_batch is not None:
        prev_pos = prev_obs_batch[:, 0]
        prev_vel = prev_obs_batch[:, 1]
        prev_angle = prev_obs_batch[:, 2]
        prev_ang_vel = prev_obs_batch[:, 3]
        expected_pos = prev_pos + prev_vel * dt
        expected_angle = prev_angle + prev_ang_vel * dt
        physics_error = (np.abs(cart_pos - expected_pos) +
                         np.abs(pole_angle - expected_angle) * 10)
        features[:, 17] = np.tanh(physics_error * 5)
    else:
        features[:, 17] = 0.5

    return features


def compute_controller_features(obs, age, target, last_action, max_age):
    """Compute features for controller input."""
    feats = np.zeros(8, dtype=np.float32)
    feats[0:4] = obs
    feats[4] = min(age, max_age) / max(max_age, 1)
    feats[5] = target
    feats[6] = target - obs[0]
    feats[7] = float(last_action)
    return feats


def compute_controller_features_batch(obs_batch, age_batch, target_batch, last_action_batch, max_age):
    """Vectorized controller features computation."""
    batch_size = obs_batch.shape[0]
    feats = np.zeros((batch_size, 8), dtype=np.float32)
    feats[:, 0:4] = obs_batch
    feats[:, 4] = np.minimum(age_batch, max_age) / max(max_age, 1)
    feats[:, 5] = target_batch
    feats[:, 6] = target_batch - obs_batch[:, 0]
    feats[:, 7] = last_action_batch.astype(np.float32)
    return feats


def prepare_feature_sequence(history, memory_length, device):
    """Prepare feature sequence for RNN input."""
    feat_seq = np.zeros((1, memory_length, 8), dtype=np.float32)
    hist_len = min(len(history), memory_length)
    if hist_len > 0:
        feat_seq[0, -hist_len:, :] = np.array(history[-hist_len:])
    return torch.as_tensor(feat_seq, dtype=torch.float32, device=device)


def prepare_feature_sequence_batch(histories, memory_length, device):
    """Vectorized feature sequence preparation."""
    batch_size = len(histories)
    feat_seq = np.zeros((batch_size, memory_length, 8), dtype=np.float32)
    for i, history in enumerate(histories):
        hist_len = min(len(history), memory_length)
        if hist_len > 0:
            feat_seq[i, -hist_len:, :] = np.array(history[-hist_len:])
    return torch.as_tensor(feat_seq, dtype=torch.float32, device=device)


# =================================================================
# 4. HELPER FUNCTIONS
# =================================================================
def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def sample_noise_std():
    return np.random.choice(CONFIG["NOISE_STD_VALUES"])


def sample_noise_std_batch(batch_size):
    return np.random.choice(CONFIG["NOISE_STD_VALUES"], size=batch_size)


def generate_noisy_observation(true_state, noise_std):
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, true_state.shape).astype(np.float32)
        return true_state + noise, noise
    return true_state.copy(), np.zeros_like(true_state)


def generate_noisy_observation_batch(true_states, noise_stds):
    """Vectorized noisy observation generation."""
    batch_size = true_states.shape[0]
    noise = np.zeros_like(true_states)
    for i in range(batch_size):
        if noise_stds[i] > 0:
            noise[i] = np.random.normal(0, noise_stds[i], true_states[i].shape)
    return true_states + noise


def save_config(config, path):
    config_serializable = {}
    for k, v in config.items():
        if isinstance(v, np.ndarray):
            config_serializable[k] = v.tolist()
        else:
            config_serializable[k] = v
    with open(path, 'w') as f:
        json.dump(config_serializable, f, indent=2)


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


def print_crash_behavior_info():
    """Print information about the current crash behavior setting."""
    print("\n" + "=" * 70)
    print("CRASH BEHAVIOR CONFIGURATION")
    print("=" * 70)

    if CONFIG["END_ON_CRASH"]:
        print(f"  Mode: END ON CRASH")
        print(f"  - Rollout/Episode terminates immediately when crash occurs")
        print(f"  - Crash reward: {CONFIG['CRASH_TERMINAL_REWARD']}")
    else:
        print(f"  Mode: CONTINUE AFTER CRASH")
        print(f"  - Rollout/Episode continues after crash (environment resets)")
        print(f"  - Crash penalty: {CONFIG['CRASH_PENALTY']}")

    print("=" * 70 + "\n")



# =================================================================
# 5. EXPIRATION MEASUREMENT
# =================================================================
def measure_expiration_single_trajectory_vectorized(controller, start_states, stale_obs_batch,
                                                     target_starts, epsilon_values, device):
    """Single trajectory expiration measurement using SyncVectorEnv.
    
    Note: Uses SyncVectorEnv instead of AsyncVectorEnv because we need to set
    specific initial states, which requires access to the underlying environments.
    """
    batch_size = start_states.shape[0]
    K = CONFIG["MAX_EXPIRATION"]
    max_age = CONFIG["MAX_AGE"]
    memory_length = CONFIG["MEMORY_LENGTH"]
    
    end_on_crash = CONFIG["END_ON_CRASH"]
    crash_terminal_reward = CONFIG["CRASH_TERMINAL_REWARD"]
    
    # Create SyncVectorEnv (allows access to underlying envs for state setting)
    envs = SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(batch_size)])
    
    try:
        # Reset all environments
        observations, infos = envs.reset()
        
        # Set initial states for each environment
        for i in range(batch_size):
            envs.envs[i].unwrapped.state = start_states[i].copy()
        
        targets = target_starts.copy()
        target_procs = []
        for i in range(batch_size):
            tp = RandomWalkTarget()
            tp.T = targets[i]
            target_procs.append(tp)
        
        hidden = torch.zeros(1, batch_size, CONFIG["RNN_HIDDEN"], device=device)
        histories = [[] for _ in range(batch_size)]
        last_actions = np.zeros(batch_size, dtype=np.int32)
        
        stale_obs = stale_obs_batch.copy()
        per_step_rewards = np.zeros((batch_size, K + 1), dtype=np.float32)
        active = np.ones(batch_size, dtype=bool)
        crash_step = np.full(batch_size, -1, dtype=np.int32)
        
        for step in range(K + 1):
            if not np.any(active):
                break
            
            current_age = np.full(batch_size, step, dtype=np.int32)
            
            controller_features = compute_controller_features_batch(
                stale_obs, current_age, targets, last_actions, max_age
            )
            
            for i in range(batch_size):
                if active[i]:
                    histories[i].append(controller_features[i])
                    if len(histories[i]) > memory_length:
                        histories[i].pop(0)
            
            feat_tensor = prepare_feature_sequence_batch(histories, memory_length, device)
            actions_t, hidden = controller.act_deterministic(feat_tensor, hidden)
            actions = actions_t.cpu().numpy()
            
            # Step ALL environments at once (VECTORIZED!)
            next_observations, rewards, terminateds, truncateds, infos = envs.step(actions)
            
            # Process results for each environment
            for i in range(batch_size):
                if not active[i]:
                    per_step_rewards[i, step] = crash_terminal_reward
                    continue
                
                if terminateds[i]:
                    per_step_rewards[i, step] = crash_terminal_reward
                    active[i] = False
                    crash_step[i] = step
                    for remaining_step in range(step + 1, K + 1):
                        per_step_rewards[i, remaining_step] = crash_terminal_reward
                else:
                    # Get true state from the environment
                    new_true_state = np.array(envs.envs[i].unwrapped.state, dtype=np.float32)
                    targets[i] = target_procs[i].step(new_true_state[0])
                    tracking_reward = np.clip(1.0 - abs(new_true_state[0] - targets[i]), -1.0, 1.0)
                    per_step_rewards[i, step] = tracking_reward
                    last_actions[i] = actions[i]
        
        num_epsilon = len(epsilon_values)
        expirations = np.zeros((batch_size, num_epsilon), dtype=np.float32)
        
        r_0 = per_step_rewards[:, 0]
        
        for eps_idx, epsilon in enumerate(epsilon_values):
            for b in range(batch_size):
                tau_star = 0
                for k in range(1, K + 1):
                    reward_degradation = r_0[b] - per_step_rewards[b, k]
                    if np.abs(reward_degradation) <= epsilon:
                        tau_star = k
                expirations[b, eps_idx] = tau_star
        
        return expirations, per_step_rewards, crash_step
    
    finally:
        envs.close()
# =================================================================
# 6. DATA COLLECTION WITH ENHANCED STATISTICS
# =================================================================
def collect_training_data_vectorized(controller, device, force_recollect=False):
    """Collect training data with tqdm progress bars."""
    dataset_path = CONFIG["DATASET_SAVE_PATH"]
    
    if os.path.exists(dataset_path) and not force_recollect:
        pp.section("Loading Existing Dataset")
        data = np.load(dataset_path, allow_pickle=True)
        features = data['features']
        expirations = data['expirations']
        noise_stds = data.get('noise_stds', None)
        
        expiration_details = None
        if 'expiration_details' in data.files:
            expiration_details = data['expiration_details'].item()
        
        pp.success(f"Loaded {len(features):,} samples")
        pp.keyvalue("Features shape", features.shape)
        return features, expirations, noise_stds, expiration_details
    
    pp.header("DATA COLLECTION")
    pp.keyvalue("Method", "Single-Trajectory Expiration Measurement")
    pp.keyvalue("Target samples", f"{CONFIG['NUM_TOTAL_SAMPLES']:,}")
    pp.keyvalue("Batch size", CONFIG['BATCH_SIZE_COLLECTION'])
    pp.keyvalue("Parallel envs", "AsyncVectorEnv")
    
    epsilon_values = CONFIG["EPSILON_VALUES"]
    total_samples = CONFIG["NUM_TOTAL_SAMPLES"]
    batch_size = CONFIG["BATCH_SIZE_COLLECTION"]
    
    all_features = []
    all_noise_stds = []
    all_expirations = []
    all_per_step_rewards = []
    
    epsilon_stats = {eps_idx: [] for eps_idx in range(len(epsilon_values))}
    noise_stats = defaultdict(list)
    per_step_reward_samples = []
    
    # Create AsyncVectorEnv for main environments
    main_envs = AsyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(batch_size)])
    
    try:
        main_targets = [RandomWalkTarget() for _ in range(batch_size)]
        main_hiddens = torch.zeros(1, batch_size, CONFIG["RNN_HIDDEN"], device=device)
        main_histories = [[] for _ in range(batch_size)]
        main_last_actions = np.zeros(batch_size, dtype=np.int32)
        main_prev_obs = [None] * batch_size
        
        observations, infos = main_envs.reset()
        main_states = observations.astype(np.float32)
        main_target_vals = np.zeros(batch_size, dtype=np.float32)
        
        for i in range(batch_size):
            main_target_vals[i] = main_targets[i].reset(main_states[i, 0])
        
        # Main progress bar
        pbar = tqdm(
            total=total_samples,
            desc="Collecting data",
            unit="samples",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
            ncols=100
        )
        
        sample_count = 0
        last_stats_update = 0
        stats_update_interval = total_samples // 10  # Update stats ~10 times
        
        while sample_count < total_samples:
            current_batch_size = min(batch_size, total_samples - sample_count)
            
            current_states = main_states[:current_batch_size].copy()
            
            noise_stds = sample_noise_std_batch(current_batch_size)
            noisy_obs = generate_noisy_observation_batch(current_states, noise_stds)
            
            prev_obs_batch = np.array([main_prev_obs[i] if main_prev_obs[i] is not None 
                                        else np.zeros(4, dtype=np.float32) 
                                        for i in range(current_batch_size)])
            has_prev = np.array([main_prev_obs[i] is not None for i in range(current_batch_size)])
            
            pred_features = compute_predictor_features_v3_batch(
                noisy_obs, 
                main_target_vals[:current_batch_size],
                prev_obs_batch if np.any(has_prev) else None
            )
            
            expirations, per_step_rewards, crash_steps = measure_expiration_single_trajectory_vectorized(
                controller=controller,
                start_states=current_states.copy(),
                stale_obs_batch=noisy_obs.copy(),
                target_starts=main_target_vals[:current_batch_size].copy(),
                epsilon_values=epsilon_values,
                device=device
            )
            
            for i in range(current_batch_size):
                all_features.append(pred_features[i])
                all_noise_stds.append(noise_stds[i])
                all_expirations.append(expirations[i])
                all_per_step_rewards.append(per_step_rewards[i])
                
                for eps_idx in range(len(epsilon_values)):
                    epsilon_stats[eps_idx].append(expirations[i, eps_idx])
                
                noise_stats[noise_stds[i]].append(expirations[i].mean())
                
                if len(per_step_reward_samples) < 1000:
                    per_step_reward_samples.append({
                        'noise_std': noise_stds[i],
                        'expirations': expirations[i].copy(),
                        'per_step_rewards': per_step_rewards[i].copy(),
                        'crash_step': crash_steps[i]
                    })
            
            sample_count += current_batch_size
            pbar.update(current_batch_size)
            
            # Update postfix with current stats
            if sample_count - last_stats_update >= stats_update_interval:
                stats = print_expiration_statistics(epsilon_values, epsilon_stats, noise_stats, sample_count, compact=True)
                if stats:
                    pbar.set_postfix({
                        'τ_mean': f"{stats['mean']:.2f}",
                        'τ_std': f"{stats['std']:.2f}"
                    })
                last_stats_update = sample_count
            
            # Advance main environments
            controller_features = compute_controller_features_batch(
                noisy_obs, np.zeros(current_batch_size, dtype=np.int32),
                main_target_vals[:current_batch_size], main_last_actions[:current_batch_size], 
                CONFIG["MAX_AGE"]
            )
            
            for i in range(current_batch_size):
                main_histories[i].append(controller_features[i])
                if len(main_histories[i]) > CONFIG["MEMORY_LENGTH"]:
                    main_histories[i].pop(0)
            
            feat_tensor = prepare_feature_sequence_batch(
                main_histories[:current_batch_size], CONFIG["MEMORY_LENGTH"], device
            )
            actions_t, main_hiddens = controller.act_deterministic(
                feat_tensor, main_hiddens[:, :current_batch_size, :]
            )
            actions = actions_t.cpu().numpy()
            
            if current_batch_size < batch_size:
                full_actions = np.zeros(batch_size, dtype=np.int32)
                full_actions[:current_batch_size] = actions
                actions = full_actions
            
            next_observations, rewards, terminateds, truncateds, infos = main_envs.step(actions)
            main_states = next_observations.astype(np.float32)
            
            for i in range(current_batch_size):
                if terminateds[i]:
                    main_target_vals[i] = main_targets[i].reset(main_states[i, 0])
                    main_hiddens[:, i, :] = 0
                    main_histories[i] = []
                    main_last_actions[i] = 0
                    main_prev_obs[i] = None
                else:
                    main_target_vals[i] = main_targets[i].step(main_states[i, 0])
                    main_last_actions[i] = actions[i]
                    main_prev_obs[i] = noisy_obs[i].copy()
        
        pbar.close()
    
    finally:
        main_envs.close()
    
    pp.success("Data collection complete!")
    
    # Print final statistics
    print_expiration_statistics(epsilon_values, epsilon_stats, noise_stats, sample_count, compact=False)
    
    features_array = np.array(all_features, dtype=np.float32)
    noise_stds_array = np.array(all_noise_stds, dtype=np.float32)
    expirations_array = np.array(all_expirations, dtype=np.float32)
    
    expiration_details = {
        'epsilon_stats': {eps: np.array(epsilon_stats[eps_idx]) 
                          for eps_idx, eps in enumerate(epsilon_values)},
        'noise_stats': {k: np.array(v) for k, v in noise_stats.items()},
        'per_step_reward_samples': per_step_reward_samples,
        'epsilon_values': epsilon_values,
    }
    
    pp.section("Saving Dataset")
    np.savez(dataset_path,
             features=features_array,
             expirations=expirations_array,
             noise_stds=noise_stds_array,
             expiration_details=expiration_details)
    pp.success(f"Saved to {dataset_path}")
    
    return features_array, expirations_array, noise_stds_array, expiration_details
# =================================================================
# 7. TRAIN PREDICTOR
# =================================================================
def train_predictor(features, expirations, epsilon_values, device,
                    noise_stds=None, force_retrain=False):
    """Train predictor with tqdm progress bar."""
    predictor_path = CONFIG["PREDICTOR_SAVE_PATH"]

    if os.path.exists(predictor_path) and not force_retrain:
        pp.section("Loading Existing Predictor")
        checkpoint = torch.load(predictor_path, map_location=device)
        predictor = SimpleExpirationPredictor(
            input_dim=CONFIG["PREDICTOR_INPUT_DIM"],
            hidden_dim=CONFIG["PREDICTOR_HIDDEN"],
            num_layers=CONFIG["PREDICTOR_LAYERS"],
            num_epsilons=len(epsilon_values),
            dropout=CONFIG["PREDICTOR_DROPOUT"],
            max_age=CONFIG["MAX_EXPIRATION"]
        ).to(device)
        predictor.load_state_dict(checkpoint['model_state'])
        predictor.eval()
        norm_params = {'mean': checkpoint['norm_mean'], 'std': checkpoint['norm_std']}
        pp.success(f"Loaded from {predictor_path}")
        return predictor, [], [], norm_params

    pp.header("TRAINING PREDICTOR")
    
    pp.keyvalue("Input dimension", CONFIG["PREDICTOR_INPUT_DIM"])
    pp.keyvalue("Hidden dimension", CONFIG["PREDICTOR_HIDDEN"])
    pp.keyvalue("Number of layers", CONFIG["PREDICTOR_LAYERS"])
    pp.keyvalue("Number of epsilons", len(epsilon_values))
    pp.keyvalue("Learning rate", CONFIG["PREDICTOR_LR"])
    pp.keyvalue("Max epochs", CONFIG["PREDICTOR_EPOCHS"])

    X = torch.tensor(features, dtype=torch.float32)
    Y = torch.tensor(expirations, dtype=torch.float32)

    n_samples = len(X)
    n_train = int(0.8 * n_samples)
    indices = np.random.permutation(n_samples)

    X_train, X_val = X[indices[:n_train]], X[indices[n_train:]]
    Y_train, Y_val = Y[indices[:n_train]], Y[indices[n_train:]]

    pp.keyvalue("Training samples", f"{n_train:,}")
    pp.keyvalue("Validation samples", f"{n_samples - n_train:,}")

    X_mean = X_train.mean(dim=0)
    X_std = X_train.std(dim=0) + 1e-8
    X_train_norm = (X_train - X_mean) / X_std
    X_val_norm = (X_val - X_mean) / X_std

    train_dataset = TensorDataset(X_train_norm, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["PREDICTOR_BATCH_SIZE"], shuffle=True)

    predictor = SimpleExpirationPredictor(
        input_dim=CONFIG["PREDICTOR_INPUT_DIM"],
        hidden_dim=CONFIG["PREDICTOR_HIDDEN"],
        num_layers=CONFIG["PREDICTOR_LAYERS"],
        num_epsilons=len(epsilon_values),
        dropout=CONFIG["PREDICTOR_DROPOUT"],
        max_age=CONFIG["MAX_EXPIRATION"]
    ).to(device)

    optimizer = optim.AdamW(predictor.parameters(), lr=CONFIG["PREDICTOR_LR"],
                            weight_decay=CONFIG["PREDICTOR_WEIGHT_DECAY"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.SmoothL1Loss()

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    norm_params = {'mean': X_mean, 'std': X_std}

    # Training progress bar
    pbar = tqdm(
        range(CONFIG["PREDICTOR_EPOCHS"]),
        desc="Training",
        unit="epoch",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
        ncols=100
    )

    for epoch in pbar:
        predictor.train()
        epoch_loss = 0.0

        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            optimizer.zero_grad()
            predictions = predictor(batch_X)
            loss = criterion(predictions, batch_Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)

        predictor.eval()
        with torch.no_grad():
            val_pred = predictor(X_val_norm.to(device))
            val_loss = criterion(val_pred, Y_val.to(device)).item()
            val_losses.append(val_loss)

        scheduler.step(val_loss)

        # Update progress bar
        pbar.set_postfix({
            'train': f'{train_loss:.4f}',
            'val': f'{val_loss:.4f}',
            'best': f'{best_val_loss:.4f}',
            'patience': f'{patience_counter}/{CONFIG["EARLY_STOPPING_PATIENCE"]}'
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = predictor.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= CONFIG["EARLY_STOPPING_PATIENCE"]:
            pbar.close()
            pp.info(f"Early stopping at epoch {epoch + 1}")
            break

    predictor.load_state_dict(best_state)
    predictor.eval()

    # Save model
    pp.section("Saving Model")
    torch.save({'model_state': predictor.state_dict(), 'norm_mean': X_mean, 'norm_std': X_std},
               predictor_path)
    pp.success(f"Saved to {predictor_path}")
    
    # Print final stats
    pp.progress_box("Training Complete", {
        'Best validation loss': best_val_loss,
        'Final training loss': train_losses[-1],
        'Epochs trained': len(train_losses),
        'Early stopped': patience_counter >= CONFIG["EARLY_STOPPING_PATIENCE"]
    })

    return predictor, train_losses, val_losses, norm_params
# =================================================================
# 8. VECTORIZED GAME FUNCTIONS WITH DETAILED METRICS
# =================================================================
def run_smart_transmission_game_vectorized(controller, predictor, epsilon_idx, epsilon_value,
                                           comm_cost, device, norm_params, num_parallel_envs=10,
                                           random_seed=None, track_pattern=False):
    """Optimized smart transmission game - reuses environments across batches."""
    if random_seed is not None:
        np.random.seed(random_seed)

    max_steps = CONFIG["MAX_EPISODE_STEPS"]
    max_age = CONFIG["MAX_AGE"]
    memory_length = CONFIG["MEMORY_LENGTH"]
    num_episodes = CONFIG["GAME_EPISODES"]

    end_on_crash = CONFIG["END_ON_CRASH"]
    crash_terminal_reward = CONFIG["CRASH_TERMINAL_REWARD"]
    crash_penalty = CONFIG["CRASH_PENALTY"]

    norm_mean = norm_params['mean'].to(device)
    norm_std = norm_params['std'].to(device)

    all_episode_data = []
    transmission_patterns = [] if track_pattern else None

    # Create environments ONCE and reuse
    batch_size = min(num_parallel_envs, num_episodes)
    envs = SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(batch_size)])

    try:
        episodes_completed = 0
        
        while episodes_completed < num_episodes:
            current_batch = min(batch_size, num_episodes - episodes_completed)
            
            target_procs = [RandomWalkTarget() for _ in range(current_batch)]

            # Reset environments
            observations, infos = envs.reset()
            true_states = observations[:current_batch].astype(np.float32)
            targets = np.zeros(current_batch, dtype=np.float32)

            for i in range(current_batch):
                targets[i] = target_procs[i].reset(true_states[i, 0])

            hidden = torch.zeros(1, current_batch, CONFIG["RNN_HIDDEN"], device=device)
            histories = [[] for _ in range(current_batch)]
            last_actions = np.zeros(current_batch, dtype=np.int32)

            noise_stds = np.random.choice(CONFIG["NOISE_STD_VALUES"], size=current_batch)
            sensor_obs = generate_noisy_observation_batch(true_states, noise_stds)

            prev_noisy_obs = [None] * current_batch

            pred_features = compute_predictor_features_v3_batch(sensor_obs, targets, None)
            feat_tensor = torch.tensor(pred_features, dtype=torch.float32).to(device)
            feat_tensor = (feat_tensor - norm_mean) / norm_std

            with torch.no_grad():
                initial_taus = predictor(feat_tensor, epsilon_idx).cpu().numpy()
                initial_taus = np.clip(np.round(initial_taus), 0, max_age).astype(int)

            tau_sensor = initial_taus.copy()
            controller_obs = sensor_obs.copy()
            controller_age = np.zeros(current_batch, dtype=np.int32)
            tau_controller = initial_taus.copy()

            transmission_counts = np.ones(current_batch, dtype=np.int32)
            cumulative_rewards = np.zeros(current_batch, dtype=np.float32)
            tracking_rewards = np.zeros(current_batch, dtype=np.float32)
            comm_costs_total = np.zeros(current_batch, dtype=np.float32)
            age_sum = np.zeros(current_batch, dtype=np.float32)
            step_count = np.zeros(current_batch, dtype=np.int32)
            
            if track_pattern and len(transmission_patterns) < 50:
                episode_patterns = [{'transmissions': [0], 'ages': [], 'predicted_taus': [initial_taus[i]]} 
                                   for i in range(min(current_batch, 50 - len(transmission_patterns)))]
            else:
                episode_patterns = None

            if end_on_crash:
                episode_active = np.ones(current_batch, dtype=bool)
                episode_crashed = np.zeros(current_batch, dtype=bool)
                survival_steps = np.zeros(current_batch, dtype=np.int32)
            else:
                crash_counts = np.zeros(current_batch, dtype=np.int32)

            for step in range(max_steps):
                if end_on_crash and not np.any(episode_active):
                    break

                true_states = np.array([envs.envs[i].unwrapped.state 
                                        for i in range(current_batch)], dtype=np.float32)
                
                new_noise_stds = np.random.choice(CONFIG["NOISE_STD_VALUES"], size=current_batch)
                new_obs = generate_noisy_observation_batch(true_states, new_noise_stds)

                prev_obs_arr = np.array([prev_noisy_obs[i] if prev_noisy_obs[i] is not None
                                         else np.zeros(4) for i in range(current_batch)], dtype=np.float32)
                new_features = compute_predictor_features_v3_batch(new_obs, targets, prev_obs_arr)

                feat_tensor = torch.tensor(new_features, dtype=torch.float32).to(device)
                feat_tensor = (feat_tensor - norm_mean) / norm_std

                with torch.no_grad():
                    tau_new = predictor(feat_tensor, epsilon_idx).cpu().numpy()
                    tau_new = np.clip(np.round(tau_new), 0, max_age).astype(int)

                transmitted = np.zeros(current_batch, dtype=bool)
                for i in range(current_batch):
                    if end_on_crash and not episode_active[i]:
                        continue

                    tau_sensor[i] = max(0, tau_sensor[i] - 1)
                    tau_controller[i] = max(0, tau_controller[i] - 1)

                    if tau_new[i] >= tau_sensor[i] or tau_sensor[i] == 0:
                        sensor_obs[i] = new_obs[i]
                        tau_sensor[i] = tau_new[i]

                    if tau_controller[i] == 0:
                        controller_obs[i] = sensor_obs[i]
                        tau_controller[i] = tau_sensor[i]
                        controller_age[i] = 0
                        transmission_counts[i] += 1
                        tau_sensor[i] = 0
                        transmitted[i] = True
                        
                        if episode_patterns is not None and i < len(episode_patterns):
                            episode_patterns[i]['transmissions'].append(step)
                            episode_patterns[i]['predicted_taus'].append(tau_new[i])
                    else:
                        controller_age[i] += 1

                    age_sum[i] += controller_age[i]
                    step_count[i] += 1
                    
                    if episode_patterns is not None and i < len(episode_patterns):
                        episode_patterns[i]['ages'].append(controller_age[i])

                controller_features = compute_controller_features_batch(
                    controller_obs, controller_age, targets, last_actions, max_age
                )

                for i in range(current_batch):
                    if end_on_crash and not episode_active[i]:
                        continue
                    histories[i].append(controller_features[i])
                    if len(histories[i]) > memory_length:
                        histories[i].pop(0)

                feat_tensor_ctrl = prepare_feature_sequence_batch(histories, memory_length, device)
                actions_t, hidden = controller.act_deterministic(feat_tensor_ctrl, hidden)
                actions = actions_t.cpu().numpy()

                for i in range(current_batch):
                    if not (end_on_crash and not episode_active[i]):
                        prev_noisy_obs[i] = new_obs[i].copy()

                # Pad actions if batch_size > current_batch
                if current_batch < batch_size:
                    full_actions = np.zeros(batch_size, dtype=np.int32)
                    full_actions[:current_batch] = actions
                    step_actions = full_actions
                else:
                    step_actions = actions

                next_observations, rewards, terminateds, truncateds, infos = envs.step(step_actions)
                
                next_true_states = next_observations[:current_batch].astype(np.float32)

                for i in range(current_batch):
                    if end_on_crash and not episode_active[i]:
                        continue

                    if terminateds[i]:
                        if end_on_crash:
                            step_reward = crash_terminal_reward
                            if transmitted[i]:
                                step_reward -= comm_cost
                                comm_costs_total[i] += comm_cost
                            cumulative_rewards[i] += step_reward
                            episode_active[i] = False
                            episode_crashed[i] = True
                            survival_steps[i] = step
                        else:
                            crash_counts[i] += 1
                            step_reward = crash_penalty
                            if transmitted[i]:
                                step_reward -= comm_cost
                                comm_costs_total[i] += comm_cost
                            cumulative_rewards[i] += step_reward

                            targets[i] = target_procs[i].reset(next_true_states[i, 0])
                            hidden[:, i, :] = 0
                            histories[i] = []
                            last_actions[i] = 0
                            prev_noisy_obs[i] = None

                            new_noise_std = np.random.choice(CONFIG["NOISE_STD_VALUES"])
                            new_obs_single = next_true_states[i] + np.random.normal(0, new_noise_std, 4).astype(np.float32) if new_noise_std > 0 else next_true_states[i].copy()

                            new_feat = compute_predictor_features_v3(new_obs_single, targets[i], None)
                            feat_t = torch.tensor(new_feat, dtype=torch.float32).unsqueeze(0).to(device)
                            feat_t = (feat_t - norm_mean) / norm_std

                            with torch.no_grad():
                                new_tau = int(round(predictor(feat_t, epsilon_idx).item()))
                                new_tau = max(0, min(new_tau, max_age))

                            sensor_obs[i] = new_obs_single
                            tau_sensor[i] = new_tau
                            controller_obs[i] = new_obs_single
                            tau_controller[i] = new_tau
                            controller_age[i] = 0
                            transmission_counts[i] += 1
                    else:
                        if end_on_crash:
                            survival_steps[i] = step + 1

                        targets[i] = target_procs[i].step(next_true_states[i, 0])

                        tracking_reward = 1.0 - abs(next_true_states[i, 0] - targets[i])
                        tracking_rewards[i] += tracking_reward
                        
                        comm_penalty_val = comm_cost if transmitted[i] else 0.0
                        if transmitted[i]:
                            comm_costs_total[i] += comm_cost

                        cumulative_rewards[i] += tracking_reward - comm_penalty_val
                        last_actions[i] = actions[i]
                
                true_states = next_true_states

            # Record episode data
            for i in range(current_batch):
                avg_age = age_sum[i] / step_count[i] if step_count[i] > 0 else 0
                
                if end_on_crash:
                    all_episode_data.append({
                        'reward': cumulative_rewards[i],
                        'tracking_reward': tracking_rewards[i],
                        'comm_cost_total': comm_costs_total[i],
                        'transmissions': transmission_counts[i],
                        'crashed': bool(episode_crashed[i]),
                        'survival_steps': int(survival_steps[i]),
                        'survival_rate': 1.0 - float(episode_crashed[i]),
                        'avg_age': avg_age,
                        'transmission_rate': transmission_counts[i] / max(survival_steps[i], 1),
                    })
                else:
                    all_episode_data.append({
                        'reward': cumulative_rewards[i],
                        'tracking_reward': tracking_rewards[i],
                        'comm_cost_total': comm_costs_total[i],
                        'transmissions': transmission_counts[i],
                        'crashes': int(crash_counts[i]),
                        'avg_age': avg_age,
                        'transmission_rate': transmission_counts[i] / max_steps,
                    })
            
            if episode_patterns is not None:
                for pattern in episode_patterns:
                    if len(transmission_patterns) < 50:
                        transmission_patterns.append(pattern)

            episodes_completed += current_batch

    finally:
        envs.close()

    if track_pattern:
        return all_episode_data, transmission_patterns
    return all_episode_data
def run_periodic_transmission_game_vectorized(controller, period, comm_cost, device,
                                              num_parallel_envs=10, random_seed=None,
                                              track_pattern=False):
    """Optimized periodic transmission game - reuses environments across batches."""
    if random_seed is not None:
        np.random.seed(random_seed)

    max_steps = CONFIG["MAX_EPISODE_STEPS"]
    max_age = CONFIG["MAX_AGE"]
    memory_length = CONFIG["MEMORY_LENGTH"]
    num_episodes = CONFIG["GAME_EPISODES"]

    end_on_crash = CONFIG["END_ON_CRASH"]
    crash_terminal_reward = CONFIG["CRASH_TERMINAL_REWARD"]
    crash_penalty = CONFIG["CRASH_PENALTY"]

    all_episode_data = []
    transmission_patterns = [] if track_pattern else None

    # Create environments ONCE and reuse
    batch_size = min(num_parallel_envs, num_episodes)
    envs = SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(batch_size)])

    try:
        episodes_completed = 0
        
        while episodes_completed < num_episodes:
            current_batch = min(batch_size, num_episodes - episodes_completed)
            
            target_procs = [RandomWalkTarget() for _ in range(current_batch)]

            observations, infos = envs.reset()
            true_states = observations[:current_batch].astype(np.float32)
            targets = np.zeros(current_batch, dtype=np.float32)

            for i in range(current_batch):
                targets[i] = target_procs[i].reset(true_states[i, 0])

            hidden = torch.zeros(1, current_batch, CONFIG["RNN_HIDDEN"], device=device)
            histories = [[] for _ in range(current_batch)]
            last_actions = np.zeros(current_batch, dtype=np.int32)

            noise_stds = np.random.choice(CONFIG["NOISE_STD_VALUES"], size=current_batch)
            controller_obs = generate_noisy_observation_batch(true_states, noise_stds)

            current_age = np.zeros(current_batch, dtype=np.int32)
            transmission_counts = np.ones(current_batch, dtype=np.int32)
            steps_since_transmission = np.zeros(current_batch, dtype=np.int32)
            cumulative_rewards = np.zeros(current_batch, dtype=np.float32)
            tracking_rewards = np.zeros(current_batch, dtype=np.float32)
            comm_costs_total = np.zeros(current_batch, dtype=np.float32)
            age_sum = np.zeros(current_batch, dtype=np.float32)
            step_count = np.zeros(current_batch, dtype=np.int32)
            
            if track_pattern and len(transmission_patterns) < 50:
                episode_patterns = [{'transmissions': [0], 'ages': []} 
                                   for i in range(min(current_batch, 50 - len(transmission_patterns)))]
            else:
                episode_patterns = None

            if end_on_crash:
                episode_active = np.ones(current_batch, dtype=bool)
                episode_crashed = np.zeros(current_batch, dtype=bool)
                survival_steps = np.zeros(current_batch, dtype=np.int32)
            else:
                crash_counts = np.zeros(current_batch, dtype=np.int32)

            for step in range(max_steps):
                if end_on_crash and not np.any(episode_active):
                    break

                true_states = np.array([envs.envs[i].unwrapped.state 
                                        for i in range(current_batch)], dtype=np.float32)

                transmitted = np.zeros(current_batch, dtype=bool)
                for i in range(current_batch):
                    if end_on_crash and not episode_active[i]:
                        continue

                    if steps_since_transmission[i] >= period:
                        noise_std = np.random.choice(CONFIG["NOISE_STD_VALUES"])
                        if noise_std > 0:
                            controller_obs[i] = true_states[i] + np.random.normal(0, noise_std, 4).astype(np.float32)
                        else:
                            controller_obs[i] = true_states[i].copy()
                        current_age[i] = 0
                        transmission_counts[i] += 1
                        steps_since_transmission[i] = 0
                        transmitted[i] = True
                        
                        if episode_patterns is not None and i < len(episode_patterns):
                            episode_patterns[i]['transmissions'].append(step)

                    age_sum[i] += current_age[i]
                    step_count[i] += 1
                    
                    if episode_patterns is not None and i < len(episode_patterns):
                        episode_patterns[i]['ages'].append(current_age[i])

                controller_features = compute_controller_features_batch(
                    controller_obs, current_age, targets, last_actions, max_age
                )

                for i in range(current_batch):
                    if end_on_crash and not episode_active[i]:
                        continue
                    histories[i].append(controller_features[i])
                    if len(histories[i]) > memory_length:
                        histories[i].pop(0)

                feat_tensor = prepare_feature_sequence_batch(histories, memory_length, device)
                actions_t, hidden = controller.act_deterministic(feat_tensor, hidden)
                actions = actions_t.cpu().numpy()

                # Pad actions if needed
                if current_batch < batch_size:
                    full_actions = np.zeros(batch_size, dtype=np.int32)
                    full_actions[:current_batch] = actions
                    step_actions = full_actions
                else:
                    step_actions = actions

                next_observations, rewards, terminateds, truncateds, infos = envs.step(step_actions)
                
                next_true_states = next_observations[:current_batch].astype(np.float32)

                for i in range(current_batch):
                    if end_on_crash and not episode_active[i]:
                        continue

                    if terminateds[i]:
                        if end_on_crash:
                            step_reward = crash_terminal_reward
                            if transmitted[i]:
                                step_reward -= comm_cost
                                comm_costs_total[i] += comm_cost
                            cumulative_rewards[i] += step_reward
                            episode_active[i] = False
                            episode_crashed[i] = True
                            survival_steps[i] = step
                        else:
                            crash_counts[i] += 1
                            step_reward = crash_penalty
                            if transmitted[i]:
                                step_reward -= comm_cost
                                comm_costs_total[i] += comm_cost
                            cumulative_rewards[i] += step_reward

                            targets[i] = target_procs[i].reset(next_true_states[i, 0])
                            hidden[:, i, :] = 0
                            histories[i] = []
                            last_actions[i] = 0

                            noise_std = np.random.choice(CONFIG["NOISE_STD_VALUES"])
                            if noise_std > 0:
                                controller_obs[i] = next_true_states[i] + np.random.normal(0, noise_std, 4).astype(np.float32)
                            else:
                                controller_obs[i] = next_true_states[i].copy()
                            current_age[i] = 0
                            transmission_counts[i] += 1
                            steps_since_transmission[i] = 0
                    else:
                        if end_on_crash:
                            survival_steps[i] = step + 1

                        targets[i] = target_procs[i].step(next_true_states[i, 0])

                        tracking_reward = 1.0 - abs(next_true_states[i, 0] - targets[i])
                        tracking_rewards[i] += tracking_reward
                        
                        comm_penalty_val = comm_cost if transmitted[i] else 0.0
                        if transmitted[i]:
                            comm_costs_total[i] += comm_cost

                        cumulative_rewards[i] += tracking_reward - comm_penalty_val
                        last_actions[i] = actions[i]
                        current_age[i] += 1
                        steps_since_transmission[i] += 1
                
                true_states = next_true_states

            # Record episode data
            for i in range(current_batch):
                avg_age = age_sum[i] / step_count[i] if step_count[i] > 0 else 0
                
                if end_on_crash:
                    all_episode_data.append({
                        'reward': cumulative_rewards[i],
                        'tracking_reward': tracking_rewards[i],
                        'comm_cost_total': comm_costs_total[i],
                        'transmissions': transmission_counts[i],
                        'crashed': bool(episode_crashed[i]),
                        'survival_steps': int(survival_steps[i]),
                        'survival_rate': 1.0 - float(episode_crashed[i]),
                        'avg_age': avg_age,
                        'transmission_rate': transmission_counts[i] / max(survival_steps[i], 1),
                    })
                else:
                    all_episode_data.append({
                        'reward': cumulative_rewards[i],
                        'tracking_reward': tracking_rewards[i],
                        'comm_cost_total': comm_costs_total[i],
                        'transmissions': transmission_counts[i],
                        'crashes': int(crash_counts[i]),
                        'avg_age': avg_age,
                        'transmission_rate': transmission_counts[i] / max_steps,
                    })
            
            if episode_patterns is not None:
                for pattern in episode_patterns:
                    if len(transmission_patterns) < 50:
                        transmission_patterns.append(pattern)

            episodes_completed += current_batch

    finally:
        envs.close()

    if track_pattern:
        return all_episode_data, transmission_patterns
    return all_episode_data
# 9. COMPREHENSIVE PLOTTING AND ANALYSIS
# =================================================================
def plot_expiration_distribution(expiration_details, save_dir):
    """Plot the distribution of expiration times for each epsilon value."""
    epsilon_values = expiration_details['epsilon_values']
    epsilon_stats = expiration_details['epsilon_stats']
    
    # Figure 1: Violin plot of expiration distribution
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Prepare data for violin plot
    exp_data = []
    positions = []
    for i, eps in enumerate(epsilon_values):
        data = epsilon_stats[eps]
        if len(data) > 0:
            exp_data.append(data)
            positions.append(i)
    
    # Violin plot
    ax = axes[0]
    parts = ax.violinplot(exp_data, positions=positions, showmeans=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor('steelblue')
        pc.set_alpha(0.7)
    parts['cmeans'].set_color('red')
    parts['cmedians'].set_color('orange')
    
    ax.set_xticks(positions)
    ax.set_xticklabels([f'{eps:.3f}' for eps in epsilon_values], rotation=45, ha='right')
    ax.set_xlabel('Epsilon (ε)', fontsize=12)
    ax.set_ylabel('Expiration Time (τ*)', fontsize=12)
    ax.set_title('Distribution of Expiration Times by Epsilon\n(Red=Mean, Orange=Median)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Box plot
    ax = axes[1]
    bp = ax.boxplot(exp_data, positions=positions, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    ax.set_xticks(positions)
    ax.set_xticklabels([f'{eps:.3f}' for eps in epsilon_values], rotation=45, ha='right')
    ax.set_xlabel('Epsilon (ε)', fontsize=12)
    ax.set_ylabel('Expiration Time (τ*)', fontsize=12)
    ax.set_title('Box Plot of Expiration Times by Epsilon', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig_expiration_distribution.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/fig_expiration_distribution.pdf", bbox_inches='tight')
    plt.close()
    print("✓ Saved: Expiration Distribution Plot")
    
    # Figure 2: Mean expiration with confidence intervals
    fig, ax = plt.subplots(figsize=(12, 6))
    
    means = []
    stds = []
    ci_lower = []
    ci_upper = []
    
    for eps in epsilon_values:
        data = epsilon_stats[eps]
        if len(data) > 0:
            means.append(np.mean(data))
            stds.append(np.std(data))
            # 95% confidence interval
            se = stats.sem(data)
            ci = se * 1.96
            ci_lower.append(np.mean(data) - ci)
            ci_upper.append(np.mean(data) + ci)
        else:
            means.append(0)
            stds.append(0)
            ci_lower.append(0)
            ci_upper.append(0)
    
    ax.plot(epsilon_values, means, 'b-o', linewidth=2, markersize=8, label='Mean τ*')
    ax.fill_between(epsilon_values, ci_lower, ci_upper, alpha=0.3, color='blue', label='95% CI')
    ax.errorbar(epsilon_values, means, yerr=stds, fmt='none', ecolor='gray', 
                capsize=3, alpha=0.5, label='±1 Std')
    
    ax.set_xlabel('Epsilon (ε)', fontsize=14)
    ax.set_ylabel('Expiration Time (τ*)', fontsize=14)
    ax.set_title('Mean Expiration Time vs Epsilon with Confidence Intervals', fontsize=16)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig_expiration_vs_epsilon.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/fig_expiration_vs_epsilon.pdf", bbox_inches='tight')
    plt.close()
    print("✓ Saved: Expiration vs Epsilon Plot")
    
    # Figure 3: Histogram grid for each epsilon
    n_eps = len(epsilon_values)
    n_cols = min(4, n_eps)
    n_rows = (n_eps + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    axes = axes.flatten() if n_eps > 1 else [axes]
    
    for i, eps in enumerate(epsilon_values):
        ax = axes[i]
        data = epsilon_stats[eps]
        if len(data) > 0:
            bins = np.arange(-0.5, CONFIG["MAX_EXPIRATION"] + 1.5, 1)
            ax.hist(data, bins=bins, density=True, alpha=0.7, color='steelblue', edgecolor='black')
            ax.axvline(np.mean(data), color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(data):.2f}')
            ax.set_xlabel('τ*')
            ax.set_ylabel('Density')
            ax.set_title(f'ε = {eps:.4f}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Expiration Time Distribution for Each Epsilon', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig_expiration_histograms.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/fig_expiration_histograms.pdf", bbox_inches='tight')
    plt.close()
    print("✓ Saved: Expiration Histograms")


def plot_expiration_vs_noise(expiration_details, save_dir):
    """Plot how noise level affects expiration time."""
    noise_stats = expiration_details['noise_stats']
    per_step_samples = expiration_details['per_step_reward_samples']
    epsilon_values = expiration_details['epsilon_values']
    
    # Figure 1: Mean expiration vs noise level
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    noise_levels = sorted(noise_stats.keys())
    mean_exps = [np.mean(noise_stats[n]) for n in noise_levels]
    std_exps = [np.std(noise_stats[n]) for n in noise_levels]
    
    ax.errorbar(noise_levels, mean_exps, yerr=std_exps, fmt='o-', linewidth=2, 
                markersize=8, capsize=5, color='steelblue')
    ax.set_xlabel('Noise Standard Deviation (σ)', fontsize=14)
    ax.set_ylabel('Mean Expiration Time (τ*)', fontsize=14)
    ax.set_title('Expiration Time vs Noise Level\n(Averaged across all epsilons)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    if len(noise_levels) > 1:
        z = np.polyfit(noise_levels, mean_exps, 1)
        p = np.poly1d(z)
        ax.plot(noise_levels, p(noise_levels), 'r--', alpha=0.7, 
                label=f'Trend: {z[0]:.2f}σ + {z[1]:.2f}')
        ax.legend()
    
    # Figure 2: Scatter plot of individual samples
    ax = axes[1]
    if len(per_step_samples) > 0:
        noise_vals = [s['noise_std'] for s in per_step_samples]
        # Use mean expiration across epsilons
        mean_exp_vals = [np.mean(s['expirations']) for s in per_step_samples]
        
        scatter = ax.scatter(noise_vals, mean_exp_vals, alpha=0.3, s=10, c='steelblue')
        ax.set_xlabel('Noise Standard Deviation (σ)', fontsize=14)
        ax.set_ylabel('Mean Expiration Time (τ*)', fontsize=14)
        ax.set_title('Individual Sample Expirations vs Noise', fontsize=14)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig_expiration_vs_noise.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/fig_expiration_vs_noise.pdf", bbox_inches='tight')
    plt.close()
    print("✓ Saved: Expiration vs Noise Plot")
    
    # Figure 3: Detailed per-epsilon expiration vs noise
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Collect per-epsilon data by noise level
    eps_noise_data = {eps_idx: defaultdict(list) for eps_idx in range(len(epsilon_values))}
    
    for sample in per_step_samples:
        noise = sample['noise_std']
        for eps_idx, exp in enumerate(sample['expirations']):
            eps_noise_data[eps_idx][noise].append(exp)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(epsilon_values)))
    
    for eps_idx, eps in enumerate(epsilon_values):
        noise_levels = sorted(eps_noise_data[eps_idx].keys())
        means = [np.mean(eps_noise_data[eps_idx][n]) for n in noise_levels]
        ax.plot(noise_levels, means, 'o-', color=colors[eps_idx], 
                linewidth=1.5, markersize=6, alpha=0.8, label=f'ε={eps:.3f}')
    
    ax.set_xlabel('Noise Standard Deviation (σ)', fontsize=14)
    ax.set_ylabel('Mean Expiration Time (τ*)', fontsize=14)
    ax.set_title('Expiration Time vs Noise Level for Each Epsilon', fontsize=14)
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig_expiration_vs_noise_per_epsilon.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/fig_expiration_vs_noise_per_epsilon.pdf", bbox_inches='tight')
    plt.close()
    print("✓ Saved: Per-Epsilon Expiration vs Noise Plot")


def plot_sample_expiration_determination(expiration_details, save_dir, num_samples=6):
    """Visualize how expiration time is determined from reward trajectory."""
    per_step_samples = expiration_details['per_step_reward_samples']
    epsilon_values = expiration_details['epsilon_values']
    
    # Select diverse samples (different noise levels and outcomes)
    noise_levels = set(s['noise_std'] for s in per_step_samples)
    selected_samples = []
    
    # Get samples with different noise levels
    for noise in sorted(noise_levels)[:num_samples]:
        samples_at_noise = [s for s in per_step_samples if s['noise_std'] == noise]
        if samples_at_noise:
            selected_samples.append(samples_at_noise[0])
    
    if len(selected_samples) < num_samples:
        # Add more samples
        remaining = num_samples - len(selected_samples)
        for s in per_step_samples[:remaining]:
            if s not in selected_samples:
                selected_samples.append(s)
    
    # Create figure
    n_cols = min(3, len(selected_samples))
    n_rows = (len(selected_samples) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if len(selected_samples) == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in (row if hasattr(row, '__iter__') else [row])]
    
    for idx, sample in enumerate(selected_samples[:len(axes_flat)]):
        ax = axes_flat[idx]
        
        rewards = sample['per_step_rewards']
        expirations = sample['expirations']
        noise = sample['noise_std']
        crash_step = sample['crash_step']
        
        K = len(rewards) - 1
        steps = np.arange(K + 1)
        
        # Plot reward trajectory
        ax.plot(steps, rewards, 'b-o', linewidth=2, markersize=6, label='Reward r(k)')
        
        r_0 = rewards[0]
        ax.axhline(y=r_0, color='green', linestyle='--', alpha=0.5, label=f'r(0)={r_0:.3f}')
        
        # Show epsilon bands and mark expiration for selected epsilons
        selected_eps_indices = [0, len(epsilon_values)//2, -1]  # First, middle, last
        colors = ['red', 'orange', 'purple']
        
        for i, eps_idx in enumerate(selected_eps_indices):
            if eps_idx >= len(epsilon_values):
                continue
            eps = epsilon_values[eps_idx]
            tau = int(expirations[eps_idx])
            
            # Draw epsilon band
            ax.fill_between(steps, r_0 - eps, r_0 + eps, alpha=0.1, color=colors[i])
            ax.axhline(y=r_0 + eps, color=colors[i], linestyle=':', alpha=0.5)
            ax.axhline(y=r_0 - eps, color=colors[i], linestyle=':', alpha=0.5)
            
            # Mark expiration point
            if tau < len(rewards):
                ax.axvline(x=tau, color=colors[i], linestyle='--', alpha=0.7)
                ax.scatter([tau], [rewards[tau]], color=colors[i], s=100, zorder=10,
                          marker='*', label=f'τ*(ε={eps:.3f})={tau}')
        
        # Mark crash if occurred
        if crash_step >= 0:
            ax.axvline(x=crash_step, color='black', linestyle='-', linewidth=2, alpha=0.8)
            ax.text(crash_step, ax.get_ylim()[1], 'CRASH', rotation=90, va='top', fontsize=8)
        
        ax.set_xlabel('Step (k)', fontsize=10)
        ax.set_ylabel('Reward', fontsize=10)
        ax.set_title(f'Sample {idx+1}: Noise σ={noise:.2f}', fontsize=11)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, K + 0.5)
    
    # Hide empty subplots
    for j in range(len(selected_samples), len(axes_flat)):
        axes_flat[j].set_visible(False)
    
    plt.suptitle('How Expiration Time τ* is Determined\n'
                 '(τ* = max k such that |r(0) - r(k)| ≤ ε)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig_expiration_determination.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/fig_expiration_determination.pdf", bbox_inches='tight')
    plt.close()
    print("✓ Saved: Expiration Determination Visualization")


def plot_communication_pattern(smart_patterns, periodic_patterns, optimal_epsilon, save_dir):
    """Plot communication patterns for smart agent vs periodic baseline."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: Transmission events over time for smart agent (multiple episodes)
    ax = axes[0, 0]
    max_steps = 200  # Show first 200 steps
    
    for i, pattern in enumerate(smart_patterns[:20]):  # Show 20 episodes
        trans = np.array(pattern['transmissions'])
        trans = trans[trans < max_steps]
        ax.scatter(trans, np.full_like(trans, i), s=10, alpha=0.7, c='blue')
    
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Episode', fontsize=12)
    ax.set_title(f'Smart Agent Transmission Events (ε={optimal_epsilon:.3f})', fontsize=14)
    ax.set_xlim(0, max_steps)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Transmission events for periodic baseline
    ax = axes[0, 1]
    if periodic_patterns:
        for i, pattern in enumerate(periodic_patterns[:20]):
            trans = np.array(pattern['transmissions'])
            trans = trans[trans < max_steps]
            ax.scatter(trans, np.full_like(trans, i), s=10, alpha=0.7, c='red')
    
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Episode', fontsize=12)
    ax.set_title('Periodic Baseline Transmission Events (T=3)', fontsize=14)
    ax.set_xlim(0, max_steps)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Inter-transmission intervals for smart agent
    ax = axes[1, 0]
    all_intervals = []
    for pattern in smart_patterns:
        trans = np.array(pattern['transmissions'])
        if len(trans) > 1:
            intervals = np.diff(trans)
            all_intervals.extend(intervals)
    
    if all_intervals:
        bins = np.arange(0, max(all_intervals) + 2) - 0.5
        ax.hist(all_intervals, bins=bins, density=True, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(np.mean(all_intervals), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean={np.mean(all_intervals):.2f}')
        ax.set_xlabel('Inter-transmission Interval (steps)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Smart Agent Inter-transmission Interval Distribution', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Panel 4: Average age over time
    ax = axes[1, 1]
    
    # Compute average age per step
    max_len = max(len(p['ages']) for p in smart_patterns if 'ages' in p)
    smart_ages_matrix = np.zeros((len(smart_patterns), max_len))
    smart_ages_matrix[:] = np.nan
    
    for i, pattern in enumerate(smart_patterns):
        if 'ages' in pattern and len(pattern['ages']) > 0:
            ages = pattern['ages'][:max_len]
            smart_ages_matrix[i, :len(ages)] = ages
    
    smart_mean_age = np.nanmean(smart_ages_matrix, axis=0)
    smart_std_age = np.nanstd(smart_ages_matrix, axis=0)
    steps = np.arange(len(smart_mean_age))
    
    ax.plot(steps[:max_steps], smart_mean_age[:max_steps], 'b-', linewidth=2, label='Smart Agent')
    ax.fill_between(steps[:max_steps], 
                    (smart_mean_age - smart_std_age)[:max_steps],
                    (smart_mean_age + smart_std_age)[:max_steps], 
                    alpha=0.2, color='blue')
    
    if periodic_patterns:
        periodic_ages_matrix = np.zeros((len(periodic_patterns), max_len))
        periodic_ages_matrix[:] = np.nan
        for i, pattern in enumerate(periodic_patterns):
            if 'ages' in pattern and len(pattern['ages']) > 0:
                ages = pattern['ages'][:max_len]
                periodic_ages_matrix[i, :len(ages)] = ages
        
        periodic_mean_age = np.nanmean(periodic_ages_matrix, axis=0)
        ax.plot(steps[:max_steps], periodic_mean_age[:max_steps], 'r--', linewidth=2, label='Periodic (T=3)')
    
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Information Age', fontsize=12)
    ax.set_title('Average Information Age Over Time', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_steps)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig_communication_pattern.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/fig_communication_pattern.pdf", bbox_inches='tight')
    plt.close()
    print("✓ Saved: Communication Pattern Plot")
    
    # Additional figure: Predicted tau values over time
    if smart_patterns and 'predicted_taus' in smart_patterns[0]:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, pattern in enumerate(smart_patterns[:10]):
            if 'predicted_taus' in pattern:
                taus = pattern['predicted_taus']
                trans = pattern['transmissions'][:len(taus)]
                ax.plot(trans, taus, 'o-', alpha=0.5, markersize=4)
        
        ax.set_xlabel('Step (at transmission)', fontsize=12)
        ax.set_ylabel('Predicted Expiration Time (τ)', fontsize=12)
        ax.set_title('Predicted Expiration Time at Each Transmission', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/fig_predicted_tau_over_time.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_dir}/fig_predicted_tau_over_time.pdf", bbox_inches='tight')
        plt.close()
        print("✓ Saved: Predicted Tau Over Time Plot")


def create_comprehensive_plots(results, expiration_details=None):
    """Create comprehensive comparison plots."""
    save_dir = CONFIG["SAVE_DIR"]
    comm_costs = results['comm_costs']
    periodic_intervals = results['periodic_intervals']
    epsilon_values = results['epsilon_values']
    end_on_crash = CONFIG["END_ON_CRASH"]
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plot expiration-related figures if details available
    if expiration_details is not None:
        print("\n--- Creating Expiration Analysis Plots ---")
        plot_expiration_distribution(expiration_details, save_dir)
        plot_expiration_vs_noise(expiration_details, save_dir)
        plot_sample_expiration_determination(expiration_details, save_dir)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FIGURE 1: Net Reward vs Communication Cost (Main Comparison)
    # ═══════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Smart agent with optimal epsilon
    optimal_rewards = [results['optimal_reward'][c] for c in comm_costs]
    optimal_epsilons = [results['optimal_epsilon'][c] for c in comm_costs]
    
    ax.plot(comm_costs, optimal_rewards, 'b-o', linewidth=3, markersize=10, 
            label='Smart (Best ε)', zorder=10)
    
    # Add epsilon annotations
    for i, (c, eps) in enumerate(zip(comm_costs, optimal_epsilons)):
        if i % 2 == 0:  # Annotate every other point
            ax.annotate(f'ε={eps:.3f}', (c, optimal_rewards[i]), 
                       textcoords="offset points", xytext=(0, 10), 
                       ha='center', fontsize=8, color='blue')
    
    # Periodic baselines
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(periodic_intervals)))
    for idx, period in enumerate(periodic_intervals):
        rewards = [results['periodic_results'][period].get(c, np.nan) for c in comm_costs]
        ax.plot(comm_costs, rewards, '--', color=colors[idx], linewidth=2, 
                marker='s', markersize=6, label=f'Periodic (T={period})')
    
    ax.set_xlabel('Communication Cost', fontsize=14)
    ax.set_ylabel('Net Reward', fontsize=14)
    ax.set_title('Net Reward vs Communication Cost\n(Smart Agent vs Periodic Baselines)', fontsize=16)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig1_reward_vs_cost.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/fig1_reward_vs_cost.pdf", bbox_inches='tight')
    plt.close()
    print("✓ Saved Figure 1: Net Reward vs Communication Cost")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FIGURE 2: Survival Rate Comparison (if END_ON_CRASH=True)
    # ═══════════════════════════════════════════════════════════════════════════
    if end_on_crash and 'smart_survival_rate' in results:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Smart agent survival rate
        smart_survival = [results['smart_survival_rate'][results['optimal_epsilon'][c]][c] 
                          for c in comm_costs]
        ax.plot(comm_costs, smart_survival, 'b-o', linewidth=3, markersize=10, 
                label='Smart (Best ε)', zorder=10)
        
        # Periodic baselines survival rate
        for idx, period in enumerate(periodic_intervals):
            survival = [results['periodic_survival_rate'][period].get(c, np.nan) 
                       for c in comm_costs]
            ax.plot(comm_costs, survival, '--', color=colors[idx], linewidth=2, 
                    marker='s', markersize=6, label=f'Periodic (T={period})')
        
        ax.set_xlabel('Communication Cost', fontsize=14)
        ax.set_ylabel('Survival Rate', fontsize=14)
        ax.set_title('Survival Rate vs Communication Cost', fontsize=16)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/fig2_survival_rate.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_dir}/fig2_survival_rate.pdf", bbox_inches='tight')
        plt.close()
        print("✓ Saved Figure 2: Survival Rate Comparison")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FIGURE 3: Average Survival Steps
    # ═══════════════════════════════════════════════════════════════════════════
    if end_on_crash and 'smart_survival_steps' in results:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Smart agent survival steps
        smart_steps = [results['smart_survival_steps'][results['optimal_epsilon'][c]][c] 
                       for c in comm_costs]
        ax.plot(comm_costs, smart_steps, 'b-o', linewidth=3, markersize=10, 
                label='Smart (Best ε)', zorder=10)
        
        # Periodic baselines
        for idx, period in enumerate(periodic_intervals):
            steps = [results['periodic_survival_steps'][period].get(c, np.nan) 
                    for c in comm_costs]
            ax.plot(comm_costs, steps, '--', color=colors[idx], linewidth=2, 
                    marker='s', markersize=6, label=f'Periodic (T={period})')
        
        ax.set_xlabel('Communication Cost', fontsize=14)
        ax.set_ylabel('Average Survival Steps', fontsize=14)
        ax.set_title('Average Survival Steps vs Communication Cost', fontsize=16)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/fig3_survival_steps.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_dir}/fig3_survival_steps.pdf", bbox_inches='tight')
        plt.close()
        print("✓ Saved Figure 3: Survival Steps Comparison")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FIGURE 4: Transmission Count Comparison
    # ═══════════════════════════════════════════════════════════════════════════
    if 'smart_transmissions' in results:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Smart agent transmissions
        smart_trans = [results['smart_transmissions'][results['optimal_epsilon'][c]][c] 
                       for c in comm_costs]
        ax.plot(comm_costs, smart_trans, 'b-o', linewidth=3, markersize=10, 
                label='Smart (Best ε)', zorder=10)
        
        # Periodic baselines
        for idx, period in enumerate(periodic_intervals):
            trans = [results['periodic_transmissions'][period].get(c, np.nan) 
                    for c in comm_costs]
            ax.plot(comm_costs, trans, '--', color=colors[idx], linewidth=2, 
                    marker='s', markersize=6, label=f'Periodic (T={period})')
        
        ax.set_xlabel('Communication Cost', fontsize=14)
        ax.set_ylabel('Average Transmissions per Episode', fontsize=14)
        ax.set_title('Transmission Count vs Communication Cost', fontsize=16)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/fig4_transmissions.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_dir}/fig4_transmissions.pdf", bbox_inches='tight')
        plt.close()
        print("✓ Saved Figure 4: Transmission Count Comparison")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FIGURE 5: Average Information Age
    # ═══════════════════════════════════════════════════════════════════════════
    if 'smart_avg_age' in results:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Smart agent average age
        smart_age = [results['smart_avg_age'][results['optimal_epsilon'][c]][c] 
                     for c in comm_costs]
        ax.plot(comm_costs, smart_age, 'b-o', linewidth=3, markersize=10, 
                label='Smart (Best ε)', zorder=10)
        
        # Periodic baselines
        for idx, period in enumerate(periodic_intervals):
            ages = [results['periodic_avg_age'][period].get(c, np.nan) 
                   for c in comm_costs]
            ax.plot(comm_costs, ages, '--', color=colors[idx], linewidth=2, 
                    marker='s', markersize=6, label=f'Periodic (T={period})')
        
        ax.set_xlabel('Communication Cost', fontsize=14)
        ax.set_ylabel('Average Information Age', fontsize=14)
        ax.set_title('Average Information Age vs Communication Cost', fontsize=16)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/fig5_avg_age.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_dir}/fig5_avg_age.pdf", bbox_inches='tight')
        plt.close()
        print("✓ Saved Figure 5: Average Age Comparison")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FIGURE 6: Reward Improvement Over Best Periodic Baseline
    # ═══════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(12, 8))
    
    improvements = []
    best_periodic_per_cost = []
    
    for c in comm_costs:
        smart_r = results['optimal_reward'][c]
        best_periodic = max(results['periodic_results'][p].get(c, -np.inf) 
                           for p in periodic_intervals)
        best_periodic_per_cost.append(best_periodic)
        improvement = smart_r - best_periodic
        improvements.append(improvement)
    
    bars = ax.bar(comm_costs, improvements, width=0.08, color='green', alpha=0.7, 
                  edgecolor='darkgreen', linewidth=1.5)
    
    # Color bars based on positive/negative
    for bar, imp in zip(bars, improvements):
        if imp < 0:
            bar.set_color('red')
            bar.set_edgecolor('darkred')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Communication Cost', fontsize=14)
    ax.set_ylabel('Reward Improvement', fontsize=14)
    ax.set_title('Smart Agent Improvement Over Best Periodic Baseline', fontsize=16)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig6_improvement.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/fig6_improvement.pdf", bbox_inches='tight')
    plt.close()
    print("✓ Saved Figure 6: Reward Improvement")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FIGURE 7: Epsilon Selection Heatmap
    # ═══════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create heatmap of rewards for all epsilon values
    reward_matrix = np.zeros((len(epsilon_values), len(comm_costs)))
    for i, eps in enumerate(epsilon_values):
        for j, c in enumerate(comm_costs):
            reward_matrix[i, j] = results['smart_results'][eps].get(c, np.nan)
    
    im = ax.imshow(reward_matrix, aspect='auto', cmap='viridis', origin='lower')
    
    ax.set_xticks(range(len(comm_costs)))
    ax.set_xticklabels([f'{c:.1f}' for c in comm_costs])
    ax.set_yticks(range(len(epsilon_values)))
    ax.set_yticklabels([f'{e:.3f}' for e in epsilon_values])
    
    ax.set_xlabel('Communication Cost', fontsize=14)
    ax.set_ylabel('Epsilon (ε)', fontsize=14)
    ax.set_title('Reward Heatmap: Epsilon vs Communication Cost', fontsize=16)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Net Reward', fontsize=12)
    
    # Mark optimal epsilon for each cost
    for j, c in enumerate(comm_costs):
        opt_eps = results['optimal_epsilon'][c]
        opt_idx = epsilon_values.index(opt_eps)
        ax.scatter(j, opt_idx, color='red', s=100, marker='*', edgecolors='white', linewidth=1.5)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig7_epsilon_heatmap.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/fig7_epsilon_heatmap.pdf", bbox_inches='tight')
    plt.close()
    print("✓ Saved Figure 7: Epsilon Selection Heatmap")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FIGURE 8: Optimal Epsilon vs Communication Cost
    # ═══════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(comm_costs, [results['optimal_epsilon'][c] for c in comm_costs], 
            'b-o', linewidth=3, markersize=10)
    
    ax.set_xlabel('Communication Cost', fontsize=14)
    ax.set_ylabel('Optimal Epsilon (ε*)', fontsize=14)
    ax.set_title('Optimal Epsilon Selection vs Communication Cost', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig8_optimal_epsilon.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/fig8_optimal_epsilon.pdf", bbox_inches='tight')
    plt.close()
    print("✓ Saved Figure 8: Optimal Epsilon Selection")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FIGURE 9: Multi-Panel Summary (2x2)
    # ═══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: Net Reward
    ax = axes[0, 0]
    ax.plot(comm_costs, optimal_rewards, 'b-o', linewidth=2, markersize=8, label='Smart (Best ε)')
    for idx, period in enumerate(periodic_intervals):
        rewards = [results['periodic_results'][period].get(c, np.nan) for c in comm_costs]
        ax.plot(comm_costs, rewards, '--', color=colors[idx], linewidth=1.5, 
                marker='s', markersize=5, label=f'T={period}')
    ax.set_xlabel('Communication Cost', fontsize=12)
    ax.set_ylabel('Net Reward', fontsize=12)
    ax.set_title('(a) Net Reward', fontsize=14)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Survival Rate (if available)
    ax = axes[0, 1]
    if end_on_crash and 'smart_survival_rate' in results:
        smart_survival = [results['smart_survival_rate'][results['optimal_epsilon'][c]][c] 
                          for c in comm_costs]
        ax.plot(comm_costs, smart_survival, 'b-o', linewidth=2, markersize=8, label='Smart (Best ε)')
        for idx, period in enumerate(periodic_intervals):
            survival = [results['periodic_survival_rate'][period].get(c, np.nan) 
                       for c in comm_costs]
            ax.plot(comm_costs, survival, '--', color=colors[idx], linewidth=1.5, 
                    marker='s', markersize=5, label=f'T={period}')
        ax.set_ylabel('Survival Rate', fontsize=12)
        ax.set_title('(b) Survival Rate', fontsize=14)
        ax.set_ylim(-0.05, 1.05)
    else:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=16)
        ax.set_title('(b) Survival Rate (N/A)', fontsize=14)
    ax.set_xlabel('Communication Cost', fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Transmissions
    ax = axes[1, 0]
    if 'smart_transmissions' in results:
        smart_trans = [results['smart_transmissions'][results['optimal_epsilon'][c]][c] 
                       for c in comm_costs]
        ax.plot(comm_costs, smart_trans, 'b-o', linewidth=2, markersize=8, label='Smart (Best ε)')
        for idx, period in enumerate(periodic_intervals):
            trans = [results['periodic_transmissions'][period].get(c, np.nan) 
                    for c in comm_costs]
            ax.plot(comm_costs, trans, '--', color=colors[idx], linewidth=1.5, 
                    marker='s', markersize=5, label=f'T={period}')
        ax.set_ylabel('Transmissions', fontsize=12)
        ax.set_title('(c) Transmissions per Episode', fontsize=14)
    else:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=16)
        ax.set_title('(c) Transmissions (N/A)', fontsize=14)
    ax.set_xlabel('Communication Cost', fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Improvement
    ax = axes[1, 1]
    bars = ax.bar(comm_costs, improvements, width=0.08, color='green', alpha=0.7)
    for bar, imp in zip(bars, improvements):
        if imp < 0:
            bar.set_color('red')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Communication Cost', fontsize=12)
    ax.set_ylabel('Improvement', fontsize=12)
    ax.set_title('(d) Improvement over Best Periodic', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig9_summary_panel.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/fig9_summary_panel.pdf", bbox_inches='tight')
    plt.close()
    print("✓ Saved Figure 9: Multi-Panel Summary")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FIGURE 10: Pareto Frontier (Reward vs Transmissions)
    # ═══════════════════════════════════════════════════════════════════════════
    if 'smart_transmissions' in results:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot for each communication cost
        cost_colors = plt.cm.coolwarm(np.linspace(0, 1, len(comm_costs)))
        
        for j, c in enumerate(comm_costs):
            # Smart agent
            smart_r = results['optimal_reward'][c]
            smart_t = results['smart_transmissions'][results['optimal_epsilon'][c]][c]
            ax.scatter(smart_t, smart_r, c=[cost_colors[j]], s=150, marker='o', 
                      edgecolors='black', linewidth=1.5, zorder=10,
                      label=f'Smart c={c:.1f}' if j in [0, len(comm_costs)//2, -1] else None)
            
            # Periodic baselines
            for period in periodic_intervals:
                per_r = results['periodic_results'][period].get(c, np.nan)
                per_t = results['periodic_transmissions'][period].get(c, np.nan)
                ax.scatter(per_t, per_r, c=[cost_colors[j]], s=80, marker='s', 
                          alpha=0.5, edgecolors='gray')
        
        ax.set_xlabel('Average Transmissions per Episode', fontsize=14)
        ax.set_ylabel('Net Reward', fontsize=14)
        ax.set_title('Trade-off: Reward vs Communication Usage', fontsize=16)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar for communication cost
        sm = plt.cm.ScalarMappable(cmap='coolwarm', 
                                   norm=plt.Normalize(vmin=min(comm_costs), vmax=max(comm_costs)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Communication Cost', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/fig10_pareto.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_dir}/fig10_pareto.pdf", bbox_inches='tight')
        plt.close()
        print("✓ Saved Figure 10: Pareto Frontier")
    
    # Communication pattern plot (if available in results)
    if 'smart_patterns' in results and 'periodic_patterns' in results:
        plot_communication_pattern(
            results['smart_patterns'], 
            results['periodic_patterns'],
            results['optimal_epsilon'][0.5],  # Use middle cost
            save_dir
        )
    
    print(f"\n✓ All figures saved to {save_dir}/")


    """Print detailed summary tables."""
    comm_costs = results['comm_costs']
    periodic_intervals = results['periodic_intervals']
    end_on_crash = CONFIG["END_ON_CRASH"]
    
    print("\n" + "=" * 100)
    print("DETAILED PERFORMANCE SUMMARY")
    print("=" * 100)
    
    # Table 1: Net Reward Comparison
    print("\n" + "-" * 90)
    print("TABLE 1: NET REWARD COMPARISON")
    print("-" * 90)
    
    header = f"{'Cost':>8} | {'Smart':>12} | {'Best ε':>8}"
    for p in periodic_intervals:
        header += f" | {'T='+str(p):>8}"
    header += f" | {'Improve':>10}"
    print(header)
    print("-" * 90)
    
    for c in comm_costs:
        smart_r = results['optimal_reward'][c]
        opt_eps = results['optimal_epsilon'][c]
        
        row = f"{c:>8.2f} | {smart_r:>12.1f} | {opt_eps:>8.4f}"
        
        best_periodic = -np.inf
        for p in periodic_intervals:
            per_r = results['periodic_results'][p].get(c, np.nan)
            row += f" | {per_r:>8.1f}"
            if per_r > best_periodic:
                best_periodic = per_r
        
        improvement = smart_r - best_periodic
        row += f" | {improvement:>+10.1f}"
        print(row)
    
    # Table 2: Survival Rate (if applicable)
    if end_on_crash and 'smart_survival_rate' in results:
        print("\n" + "-" * 80)
        print("TABLE 2: SURVIVAL RATE COMPARISON")
        print("-" * 80)
        
        header = f"{'Cost':>8} | {'Smart':>12}"
        for p in periodic_intervals:
            header += f" | {'T='+str(p):>8}"
        print(header)
        print("-" * 80)
        
        for c in comm_costs:
            opt_eps = results['optimal_epsilon'][c]
            smart_s = results['smart_survival_rate'][opt_eps][c]
            
            row = f"{c:>8.2f} | {smart_s:>12.2%}"
            
            for p in periodic_intervals:
                per_s = results['periodic_survival_rate'][p].get(c, np.nan)
                row += f" | {per_s:>8.2%}"
            print(row)
    
    # Table 3: Transmissions
    if 'smart_transmissions' in results:
        print("\n" + "-" * 80)
        print("TABLE 3: AVERAGE TRANSMISSIONS PER EPISODE")
        print("-" * 80)
        
        header = f"{'Cost':>8} | {'Smart':>12}"
        for p in periodic_intervals:
            header += f" | {'T='+str(p):>8}"
        print(header)
        print("-" * 80)
        
        for c in comm_costs:
            opt_eps = results['optimal_epsilon'][c]
            smart_t = results['smart_transmissions'][opt_eps][c]
            
            row = f"{c:>8.2f} | {smart_t:>12.1f}"
            
            for p in periodic_intervals:
                per_t = results['periodic_transmissions'][p].get(c, np.nan)
                row += f" | {per_t:>8.1f}"
            print(row)
    
    # Table 4: Average Information Age
    if 'smart_avg_age' in results:
        print("\n" + "-" * 80)
        print("TABLE 4: AVERAGE INFORMATION AGE")
        print("-" * 80)
        
        header = f"{'Cost':>8} | {'Smart':>12}"
        for p in periodic_intervals:
            header += f" | {'T='+str(p):>8}"
        print(header)
        print("-" * 80)
        
        for c in comm_costs:
            opt_eps = results['optimal_epsilon'][c]
            smart_age = results['smart_avg_age'][opt_eps][c]
            
            row = f"{c:>8.2f} | {smart_age:>12.2f}"
            
            for p in periodic_intervals:
                per_age = results['periodic_avg_age'][p].get(c, np.nan)
                row += f" | {per_age:>8.2f}"
            print(row)
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    # Average improvement
    improvements = []
    for c in comm_costs:
        smart_r = results['optimal_reward'][c]
        best_periodic = max(results['periodic_results'][p].get(c, -np.inf) 
                           for p in periodic_intervals)
        improvements.append(smart_r - best_periodic)
    
    print(f"\nAverage improvement over best periodic: {np.mean(improvements):+.2f}")
    print(f"Maximum improvement: {max(improvements):+.2f} (at cost {comm_costs[np.argmax(improvements)]:.1f})")
    print(f"Minimum improvement: {min(improvements):+.2f} (at cost {comm_costs[np.argmin(improvements)]:.1f})")
    
    # Costs where smart wins
    wins = sum(1 for imp in improvements if imp > 0)
    print(f"\nSmart agent wins at {wins}/{len(comm_costs)} cost levels ({100*wins/len(comm_costs):.1f}%)")
    
    # Epsilon analysis
    print("\n" + "-" * 50)
    print("OPTIMAL EPSILON ANALYSIS")
    print("-" * 50)
    opt_epsilons = [results['optimal_epsilon'][c] for c in comm_costs]
    print(f"Range of optimal ε: [{min(opt_epsilons):.4f}, {max(opt_epsilons):.4f}]")
    print(f"Mean optimal ε: {np.mean(opt_epsilons):.4f}")
    print(f"Most common optimal ε: {max(set(opt_epsilons), key=opt_epsilons.count):.4f}")
    
    print("=" * 80)

def print_detailed_summary(results):
    """Print detailed summary tables with clean formatting."""
    comm_costs = results['comm_costs']
    periodic_intervals = results['periodic_intervals']
    end_on_crash = CONFIG["END_ON_CRASH"]
    
    pp.header("DETAILED PERFORMANCE SUMMARY")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TABLE 1: NET REWARD COMPARISON
    # ═══════════════════════════════════════════════════════════════════════════
    pp.subheader("TABLE 1: NET REWARD COMPARISON")
    
    columns = ["Cost", "Smart", "Best ε"] + [f"T={p}" for p in periodic_intervals] + ["Improve"]
    widths = [8, 10, 8] + [8] * len(periodic_intervals) + [10]
    pp.table_header(columns, widths)
    
    improvements = []
    for c in comm_costs:
        smart_r = results['optimal_reward'][c]
        opt_eps = results['optimal_epsilon'][c]
        
        values = [f"{c:.2f}", f"{smart_r:.1f}", f"{opt_eps:.4f}"]
        
        best_periodic = -np.inf
        for p in periodic_intervals:
            per_r = results['periodic_results'][p].get(c, np.nan)
            values.append(f"{per_r:.1f}")
            if per_r > best_periodic:
                best_periodic = per_r
        
        improvement = smart_r - best_periodic
        improvements.append(improvement)
        values.append(f"{improvement:+.1f}")
        
        print(f"  {' │ '.join(f'{v:^{w}}' for v, w in zip(values, widths))}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TABLE 2: SURVIVAL RATE (if applicable)
    # ═══════════════════════════════════════════════════════════════════════════
    if end_on_crash and 'smart_survival_rate' in results:
        pp.subheader("TABLE 2: SURVIVAL RATE COMPARISON")
        
        columns = ["Cost", "Smart"] + [f"T={p}" for p in periodic_intervals]
        widths = [8, 10] + [8] * len(periodic_intervals)
        pp.table_header(columns, widths)
        
        for c in comm_costs:
            opt_eps = results['optimal_epsilon'][c]
            smart_s = results['smart_survival_rate'][opt_eps][c]
            
            values = [f"{c:.2f}", f"{smart_s:.1%}"]
            
            for p in periodic_intervals:
                per_s = results['periodic_survival_rate'][p].get(c, np.nan)
                values.append(f"{per_s:.1%}")
            
            print(f"  {' │ '.join(f'{v:^{w}}' for v, w in zip(values, widths))}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TABLE 3: TRANSMISSIONS
    # ═══════════════════════════════════════════════════════════════════════════
    if 'smart_transmissions' in results:
        pp.subheader("TABLE 3: AVERAGE TRANSMISSIONS PER EPISODE")
        
        columns = ["Cost", "Smart"] + [f"T={p}" for p in periodic_intervals]
        widths = [8, 10] + [8] * len(periodic_intervals)
        pp.table_header(columns, widths)
        
        for c in comm_costs:
            opt_eps = results['optimal_epsilon'][c]
            smart_t = results['smart_transmissions'][opt_eps][c]
            
            values = [f"{c:.2f}", f"{smart_t:.1f}"]
            
            for p in periodic_intervals:
                per_t = results['periodic_transmissions'][p].get(c, np.nan)
                values.append(f"{per_t:.1f}")
            
            print(f"  {' │ '.join(f'{v:^{w}}' for v, w in zip(values, widths))}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TABLE 4: AVERAGE INFORMATION AGE
    # ═══════════════════════════════════════════════════════════════════════════
    if 'smart_avg_age' in results:
        pp.subheader("TABLE 4: AVERAGE INFORMATION AGE")
        
        columns = ["Cost", "Smart"] + [f"T={p}" for p in periodic_intervals]
        widths = [8, 10] + [8] * len(periodic_intervals)
        pp.table_header(columns, widths)
        
        for c in comm_costs:
            opt_eps = results['optimal_epsilon'][c]
            smart_age = results['smart_avg_age'][opt_eps][c]
            
            values = [f"{c:.2f}", f"{smart_age:.2f}"]
            
            for p in periodic_intervals:
                per_age = results['periodic_avg_age'][p].get(c, np.nan)
                values.append(f"{per_age:.2f}")
            
            print(f"  {' │ '.join(f'{v:^{w}}' for v, w in zip(values, widths))}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SUMMARY STATISTICS BOX
    # ═══════════════════════════════════════════════════════════════════════════
    opt_epsilons = [results['optimal_epsilon'][c] for c in comm_costs]
    wins = sum(1 for imp in improvements if imp > 0)
    
    summary_lines = [
        f"Average improvement over best periodic: {np.mean(improvements):+.2f}",
        f"Maximum improvement: {max(improvements):+.2f} (at cost {comm_costs[np.argmax(improvements)]:.1f})",
        f"Minimum improvement: {min(improvements):+.2f} (at cost {comm_costs[np.argmin(improvements)]:.1f})",
        "",
        f"Smart agent wins at {wins}/{len(comm_costs)} cost levels ({100*wins/len(comm_costs):.1f}%)",
        "",
        f"Optimal ε range: [{min(opt_epsilons):.4f}, {max(opt_epsilons):.4f}]",
        f"Mean optimal ε: {np.mean(opt_epsilons):.4f}",
    ]
    
    pp.final_box("SUMMARY STATISTICS", summary_lines)
# =================================================================
# 10. MAIN ANALYSIS
# =================================================================
def run_full_analysis(force_recollect=True, force_retrain=True):
    """Main analysis with organized progress output."""
    
    pp.header("SEMANTIC COMMUNICATION EXPERIMENT")
    pp.keyvalue("Timestamp", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    pp.keyvalue("Device", CONFIG["DEVICE"])
    pp.keyvalue("Random seed", CONFIG["RANDOM_SEED"])
    
    set_random_seed(CONFIG["RANDOM_SEED"])
    device = CONFIG["DEVICE"]

    print_crash_behavior_info()

    save_config(CONFIG, CONFIG["CONFIG_SAVE_PATH"])

    # Load controller
    pp.section("Loading Controller")
    controller = RecurrentActorCritic(8, 2, CONFIG["HIDDEN"], CONFIG["DEPTH"], CONFIG["RNN_HIDDEN"]).to(device)
    controller.load_state_dict(torch.load(CONFIG["CONTROLLER_PATH"], map_location=device))
    controller.eval()
    pp.success(f"Loaded from {CONFIG['CONTROLLER_PATH']}")

    epsilon_values = CONFIG["EPSILON_VALUES"]
    comm_costs = CONFIG["COMMUNICATION_COSTS"]
    periodic_intervals = CONFIG["PERIODIC_INTERVALS"]
    end_on_crash = CONFIG["END_ON_CRASH"]

    # Data collection
    features, expirations, noise_stds, expiration_details = collect_training_data_vectorized(
        controller, device, force_recollect
    )

    # Train predictor
    predictor, train_losses, val_losses, norm_params = train_predictor(
        features, expirations, epsilon_values, device, noise_stds, force_retrain
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # SMART AGENT GAMES
    # ═══════════════════════════════════════════════════════════════════════════
    pp.header("RUNNING SMART AGENT GAMES")
    pp.keyvalue("Epsilon values", len(epsilon_values))
    pp.keyvalue("Communication costs", len(comm_costs))
    pp.keyvalue("Episodes per config", CONFIG["GAME_EPISODES"])

    smart_results = {}
    smart_transmissions = {}
    smart_avg_age = {}
    smart_patterns_collected = None
    
    if end_on_crash:
        smart_survival_rate = {}
        smart_survival_steps = {}
    else:
        smart_crashes = {}

    total_smart_configs = len(epsilon_values) * len(comm_costs)
    
    with tqdm(total=total_smart_configs, desc="Smart Agent", unit="config", ncols=100) as pbar:
        for eps_idx, eps in enumerate(epsilon_values):
            smart_results[eps] = {}
            smart_transmissions[eps] = {}
            smart_avg_age[eps] = {}
            
            if end_on_crash:
                smart_survival_rate[eps] = {}
                smart_survival_steps[eps] = {}
            else:
                smart_crashes[eps] = {}

            for cost in comm_costs:
                track_pattern = (eps_idx == len(epsilon_values) // 2 and cost == 0.5)
                
                result = run_smart_transmission_game_vectorized(
                    controller, predictor, eps_idx, eps, cost, device, norm_params,
                    num_parallel_envs=CONFIG["NUM_PARALLEL_ENVS"],
                    random_seed=CONFIG["RANDOM_SEED"] + int(cost * 100),
                    track_pattern=track_pattern
                )
                
                if track_pattern:
                    data, patterns = result
                    smart_patterns_collected = patterns
                else:
                    data = result

                smart_results[eps][cost] = np.mean([d['reward'] for d in data])
                smart_transmissions[eps][cost] = np.mean([d['transmissions'] for d in data])
                smart_avg_age[eps][cost] = np.mean([d['avg_age'] for d in data])
                
                if end_on_crash:
                    smart_survival_rate[eps][cost] = np.mean([d['survival_rate'] for d in data])
                    smart_survival_steps[eps][cost] = np.mean([d['survival_steps'] for d in data])
                else:
                    smart_crashes[eps][cost] = np.mean([d['crashes'] for d in data])

                pbar.update(1)
                pbar.set_postfix({
                    'ε': f'{eps:.3f}',
                    'c': f'{cost:.1f}',
                    'R': f'{smart_results[eps][cost]:.0f}'
                })

    # Find optimal epsilon for each cost
    optimal_epsilon = {}
    optimal_reward = {}
    for cost in comm_costs:
        best_eps = max(epsilon_values, key=lambda e: smart_results[e].get(cost, -np.inf))
        optimal_epsilon[cost] = best_eps
        optimal_reward[cost] = smart_results[best_eps][cost]

    # Print optimal epsilon summary
    pp.subheader("OPTIMAL EPSILON SELECTION")
    columns = ["Cost", "Best ε", "Reward"]
    widths = [10, 12, 12]
    pp.table_header(columns, widths)
    for cost in comm_costs:
        values = [f"{cost:.2f}", f"{optimal_epsilon[cost]:.4f}", f"{optimal_reward[cost]:.1f}"]
        print(f"  {' │ '.join(f'{v:^{w}}' for v, w in zip(values, widths))}")

    # ═══════════════════════════════════════════════════════════════════════════
    # PERIODIC BASELINE GAMES
    # ═══════════════════════════════════════════════════════════════════════════
    pp.header("RUNNING PERIODIC BASELINE GAMES")
    pp.keyvalue("Periodic intervals", periodic_intervals)
    pp.keyvalue("Communication costs", len(comm_costs))

    periodic_results = {}
    periodic_transmissions = {}
    periodic_avg_age = {}
    periodic_patterns_collected = None
    
    if end_on_crash:
        periodic_survival_rate = {}
        periodic_survival_steps = {}
    else:
        periodic_crashes = {}

    total_periodic_configs = len(periodic_intervals) * len(comm_costs)
    
    with tqdm(total=total_periodic_configs, desc="Periodic Baseline", unit="config", ncols=100) as pbar:
        for period in periodic_intervals:
            periodic_results[period] = {}
            periodic_transmissions[period] = {}
            periodic_avg_age[period] = {}
            
            if end_on_crash:
                periodic_survival_rate[period] = {}
                periodic_survival_steps[period] = {}
            else:
                periodic_crashes[period] = {}

            for cost in comm_costs:
                track_pattern = (period == 3 and cost == 0.5)
                
                result = run_periodic_transmission_game_vectorized(
                    controller, period, cost, device,
                    num_parallel_envs=CONFIG["NUM_PARALLEL_ENVS"],
                    random_seed=CONFIG["RANDOM_SEED"] + int(cost * 100),
                    track_pattern=track_pattern
                )
                
                if track_pattern:
                    data, patterns = result
                    periodic_patterns_collected = patterns
                else:
                    data = result
                
                periodic_results[period][cost] = np.mean([d['reward'] for d in data])
                periodic_transmissions[period][cost] = np.mean([d['transmissions'] for d in data])
                periodic_avg_age[period][cost] = np.mean([d['avg_age'] for d in data])
                
                if end_on_crash:
                    periodic_survival_rate[period][cost] = np.mean([d['survival_rate'] for d in data])
                    periodic_survival_steps[period][cost] = np.mean([d['survival_steps'] for d in data])
                else:
                    periodic_crashes[period][cost] = np.mean([d['crashes'] for d in data])

                pbar.update(1)
                pbar.set_postfix({
                    'T': period,
                    'c': f'{cost:.1f}',
                    'R': f'{periodic_results[period][cost]:.0f}'
                })

    # ═══════════════════════════════════════════════════════════════════════════
    # COMPILE AND SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════════════════
    pp.section("Compiling Results")
    
    results = {
        'smart_results': smart_results,
        'smart_transmissions': smart_transmissions,
        'smart_avg_age': smart_avg_age,
        'optimal_epsilon': optimal_epsilon,
        'optimal_reward': optimal_reward,
        'periodic_results': periodic_results,
        'periodic_transmissions': periodic_transmissions,
        'periodic_avg_age': periodic_avg_age,
        'epsilon_values': epsilon_values,
        'comm_costs': comm_costs,
        'periodic_intervals': periodic_intervals,
    }
    
    if smart_patterns_collected is not None:
        results['smart_patterns'] = smart_patterns_collected
    if periodic_patterns_collected is not None:
        results['periodic_patterns'] = periodic_patterns_collected
    
    if end_on_crash:
        results['smart_survival_rate'] = smart_survival_rate
        results['smart_survival_steps'] = smart_survival_steps
        results['periodic_survival_rate'] = periodic_survival_rate
        results['periodic_survival_steps'] = periodic_survival_steps
    else:
        results['smart_crashes'] = smart_crashes
        results['periodic_crashes'] = periodic_crashes
    
    results['expiration_details'] = expiration_details

    with open(CONFIG["RESULTS_SAVE_PATH"], 'wb') as f:
        pickle.dump(results, f)
    pp.success(f"Results saved to {CONFIG['RESULTS_SAVE_PATH']}")

    return results, expiration_details
# =================================================================
# MAIN
# =================================================================
if __name__ == "__main__":
    pp.header("SEMANTIC COMMUNICATION EXPERIMENT", width=80)
    pp.keyvalue("Timestamp", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    pp.keyvalue("Python version", sys.version.split()[0])
    pp.keyvalue("PyTorch version", torch.__version__)
    pp.keyvalue("Device", CONFIG["DEVICE"])

    results, expiration_details = run_full_analysis(force_recollect=True, force_retrain=True)
    
    print_detailed_summary(results)
    
    create_comprehensive_plots(results, expiration_details)
    
    if 'smart_patterns' in results and 'periodic_patterns' in results:
        plot_communication_pattern(
            results['smart_patterns'],
            results['periodic_patterns'],
            results['optimal_epsilon'].get(0.5, results['epsilon_values'][0]),
            CONFIG["SAVE_DIR"]
        )

    pp.final_box("EXPERIMENT COMPLETE", [
        f"Results directory: {CONFIG['SAVE_DIR']}",
        f"Dataset: {CONFIG['DATASET_SAVE_PATH']}",
        f"Predictor: {CONFIG['PREDICTOR_SAVE_PATH']}",
        f"Results: {CONFIG['RESULTS_SAVE_PATH']}",
    ])
