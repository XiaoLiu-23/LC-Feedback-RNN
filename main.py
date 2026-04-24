import os
import math
import csv
import copy
import random
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA


# ============================================================
# 0. Reproducibility and device
# ============================================================
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


DEFAULT_SEED = 42
set_seed(DEFAULT_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True


# ============================================================
# 1. Config
# ============================================================
@dataclass
class Config:
    threshold: float = 0.5

    train_T: int = 200
    probe_T: int = 300

    # Type A: small square wave
    sq_amp: float = 0.65
    sq_width_min: int = 6
    sq_width_max: int = 10
    sq_center_min: int = 30
    sq_center_max: int = 70

    # Type B: large triangle wave
    tri_amp_min: float = 0.85
    tri_amp_max: float = 1.5
    tri_rise: int = 10
    tri_fall: int = 30
    tri_center_min: int = 30
    tri_center_max: int = 70

    # Small output triangle (Type A response, and follow-ups after Type B)
    small_out_peak: float = 0.4
    small_out_rise: int = 10
    small_out_fall: int = 15

    # Large output triangle (Type B primary response)
    large_out_peak: float = 1.0
    large_out_rise: int = 22
    large_out_fall: int = 35

    # Damped-wave tail after the large triangle (Type B):
    # a rectified cosine whose amplitude decays exponentially.
    # Produces a series of positive humps with gradually decreasing amplitude.
    wave_peak: float = 0.7
    wave_period: int = 30
    wave_decay_tau: float = 50.0
    wave_duration: int = 80

    # Extra weight applied to the tail region of Type B targets during training.
    # Without this, plain MSE is dominated by the main bump (peak 1.0) and the
    # tail (peak ~0.5) contributes too little gradient to learn.
    tail_weight: float = 1.0

    # Teacher-forcing schedule (feedback model only). We decay the probability
    # of using the reference value as feedback over training, then spend the
    # last `tf_free_run_fraction` of epochs fully free-running (tf_ratio=0) so
    # the model is forced to drive its own post-stimulus dynamics.
    tf_start: float = 0.8
    tf_end: float = 0.05
    tf_decay: str = "quadratic"     # "linear" | "quadratic" | "cosine"
    tf_free_run_fraction: float = 0.25

    include_negative: bool = True
    negative_prob: float = 0.15

    train_samples: int = 5000
    val_samples: int = 600

    batch_size: int = 128
    epochs: int = 200
    lr: float = 1e-3

    n_hidden: int = 128
    rank: int = 3
    input_dim: int = 1
    output_dim: int = 1
    tau: float = 10.0
    dt: float = 1.0
    sigma_rec: float = 0.01

    # Multi-seed experiment
    multi_seeds: Tuple[int, ...] = (42, 123, 2024)
    multi_seed_epochs: int = 50
    multi_seed_samples: int = 3000

    base_dir: str = "results"


CFG = Config()


# ============================================================
# 2. Utilities
# ============================================================
def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def save_csv(rows: List[Dict], save_path: str) -> None:
    if not rows:
        return
    ensure_dir(os.path.dirname(save_path))
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a, b = a.reshape(-1), b.reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_output_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    p = pred.reshape(-1)
    t = target.reshape(-1)
    cos = cosine_similarity(p, t)
    mse = float(((p - t) ** 2).mean())
    if p.std() > 1e-8 and t.std() > 1e-8:
        corr = float(np.corrcoef(p, t)[0, 1])
    else:
        corr = 0.0
    return {"cosine": cos, "mse": mse, "corr": corr}


# ============================================================
# 3. Waveform builders
# ============================================================
def make_square_pulse(T: int, start: int, width: int, amp: float) -> np.ndarray:
    u = np.zeros(T, dtype=np.float32)
    s = max(0, min(T, start))
    e = max(0, min(T, start + width))
    u[s:e] = amp
    return u


def add_triangle_pulse(u: np.ndarray, center: int, baseline: float, peak: float,
                       rise_len: int, fall_len: int) -> np.ndarray:
    T = len(u)
    for i in range(rise_len):
        t = center - rise_len + i
        if 0 <= t < T:
            val = baseline + (peak - baseline) * (i + 1) / rise_len
            u[t] = max(u[t], val)
    for i in range(fall_len):
        t = center + i
        if 0 <= t < T:
            val = peak - (peak - baseline) * (i + 1) / fall_len
            u[t] = max(u[t], val)
    return u


def make_triangle_bump(T: int, start_idx: int, rise_len: int, fall_len: int,
                       peak: float) -> np.ndarray:
    y = np.zeros(T, dtype=np.float32)
    for i in range(rise_len):
        t = start_idx + i
        if 0 <= t < T:
            y[t] = peak * (i + 1) / rise_len
    for i in range(fall_len):
        t = start_idx + rise_len + i
        if 0 <= t < T:
            y[t] = peak * max(0.0, 1.0 - (i + 1) / fall_len)
    return y


def make_damped_wave(T: int, start_idx: int, duration: int, peak: float,
                     period: int, decay_tau: float) -> np.ndarray:
    """
    Rectified damped cosine: a train of positive humps whose amplitude
    decays exponentially. y(i) = peak * exp(-i/decay_tau) * (1 - cos(2*pi*i/period)) / 2.
    Starts at 0, first hump peaks at i = period / 2.
    """
    y = np.zeros(T, dtype=np.float32)
    two_pi_over_period = 2.0 * math.pi / period
    for i in range(duration):
        t = start_idx + i
        if 0 <= t < T:
            env = math.exp(-i / decay_tau)
            osc = 0.5 * (1.0 - math.cos(two_pi_over_period * i))
            y[t] = peak * env * osc
    return y


def _trigger_of_square(start: int) -> int:
    return start


def _trigger_of_triangle(center: int, baseline: float, peak: float) -> int:
    if peak <= CFG.threshold:
        return -1
    i_needed = int(np.ceil((CFG.threshold - baseline) / (peak - baseline) * CFG.tri_rise)) - 1
    return max(0, center - CFG.tri_rise + max(0, i_needed))


def _build_reference(T: int, triggers: List[Tuple[int, str]]) -> np.ndarray:
    """triggers: list of (time, 'square'|'triangle')."""
    y = np.zeros(T, dtype=np.float32)
    for trig, shape in triggers:
        if trig < 0:
            continue
        if shape == "square":
            y += make_triangle_bump(T, trig, CFG.small_out_rise, CFG.small_out_fall,
                                    CFG.small_out_peak)
        else:
            y += make_triangle_bump(T, trig, CFG.large_out_rise, CFG.large_out_fall,
                                    CFG.large_out_peak)
            large_end = trig + CFG.large_out_rise + CFG.large_out_fall
            y += make_damped_wave(T, large_end, CFG.wave_duration, CFG.wave_peak,
                                  CFG.wave_period, CFG.wave_decay_tau)
    return np.clip(y, 0.0, 1.5).astype(np.float32)[:, None]


# ============================================================
# 4. Training-sequence generators
# ============================================================
def generate_type_A_sequence(T=None, threshold=None, noise_std=0.02):
    T = T if T is not None else CFG.train_T
    threshold = threshold if threshold is not None else CFG.threshold

    baseline = np.random.uniform(0.0, 0.15)
    center = np.random.randint(CFG.sq_center_min, CFG.sq_center_max)
    width = np.random.randint(CFG.sq_width_min, CFG.sq_width_max + 1)
    start = max(0, center - width // 2)

    u = np.ones(T, dtype=np.float32) * baseline
    u = np.maximum(u, make_square_pulse(T, start, width, CFG.sq_amp))
    u += np.random.randn(T).astype(np.float32) * noise_std

    above = np.where(u > threshold)[0]
    if len(above) == 0:
        u[start] = threshold + 0.05
        above = np.where(u > threshold)[0]
    trigger_t = int(above[0])

    y = make_triangle_bump(T, trigger_t, CFG.small_out_rise, CFG.small_out_fall,
                           CFG.small_out_peak)
    return u[:, None].astype(np.float32), y[:, None].astype(np.float32), trigger_t


def generate_type_B_sequence(T=None, threshold=None, noise_std=0.02):
    T = T if T is not None else CFG.train_T
    threshold = threshold if threshold is not None else CFG.threshold

    baseline = np.random.uniform(0.0, 0.15)
    center = np.random.randint(CFG.tri_center_min, CFG.tri_center_max)
    amp = np.random.uniform(CFG.tri_amp_min, CFG.tri_amp_max)

    u = np.ones(T, dtype=np.float32) * baseline
    u = add_triangle_pulse(u, center, baseline, amp, CFG.tri_rise, CFG.tri_fall)
    u += np.random.randn(T).astype(np.float32) * noise_std

    above = np.where(u > threshold)[0]
    if len(above) == 0:
        u[center] = threshold + 0.1
        above = np.where(u > threshold)[0]
    trigger_t = int(above[0])

    y = make_triangle_bump(T, trigger_t, CFG.large_out_rise, CFG.large_out_fall,
                           CFG.large_out_peak)
    large_end = trigger_t + CFG.large_out_rise + CFG.large_out_fall
    y = y + make_damped_wave(T, large_end, CFG.wave_duration, CFG.wave_peak,
                             CFG.wave_period, CFG.wave_decay_tau)
    y = np.clip(y, 0.0, 1.5)
    return u[:, None].astype(np.float32), y[:, None].astype(np.float32), trigger_t


def generate_negative_sequence(T=None, threshold=None, noise_std=0.02):
    T = T if T is not None else CFG.train_T
    threshold = threshold if threshold is not None else CFG.threshold

    baseline = np.random.uniform(0.0, 0.15)
    center = np.random.randint(30, 80)

    u = np.ones(T, dtype=np.float32) * baseline
    amp = np.random.uniform(0.1, threshold - 0.08)

    if np.random.rand() < 0.5:
        width = np.random.randint(6, 14)
        start = max(0, center - width // 2)
        u = np.maximum(u, make_square_pulse(T, start, width, amp))
    else:
        u = add_triangle_pulse(u, center, baseline, amp, 12, 15)

    u += np.random.randn(T).astype(np.float32) * noise_std
    u = np.minimum(u, threshold - 0.03)
    y = np.zeros(T, dtype=np.float32)
    return u[:, None].astype(np.float32), y[:, None].astype(np.float32), -1


class TwoStimDataset(Dataset):
    def __init__(self, n_samples: int, T: int, threshold: float):
        inputs, targets, triggers, stim_types, weights = [], [], [], [], []
        for _ in range(n_samples):
            r = np.random.rand()
            if CFG.include_negative and r < CFG.negative_prob:
                u, y, trig = generate_negative_sequence(T=T, threshold=threshold)
                stype = 0
            elif r < CFG.negative_prob + (1 - CFG.negative_prob) * 0.5:
                u, y, trig = generate_type_A_sequence(T=T, threshold=threshold)
                stype = 1
            else:
                u, y, trig = generate_type_B_sequence(T=T, threshold=threshold)
                stype = 2

            # Per-timestep loss weight. For Type B, upweight the tail region
            # (from the end of the large triangle through the damped wave)
            # so the network pays attention to it despite its small amplitude.
            w = np.ones((T, 1), dtype=np.float32)
            if stype == 2 and trig >= 0:
                large_end = trig + CFG.large_out_rise + CFG.large_out_fall
                wave_end = min(T, large_end + CFG.wave_duration)
                if large_end < T:
                    w[large_end:wave_end, 0] = 1.0 + CFG.tail_weight

            inputs.append(u)
            targets.append(y)
            triggers.append(trig)
            stim_types.append(stype)
            weights.append(w)

        # Pre-store as tensors (avoid per-batch numpy→tensor conversion)
        self.inputs = torch.from_numpy(np.stack(inputs))
        self.targets = torch.from_numpy(np.stack(targets))
        self.weights = torch.from_numpy(np.stack(weights))
        self.triggers = np.array(triggers)
        self.stim_types = np.array(stim_types)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.targets[idx], self.weights[idx]


# ============================================================
# 5. Probe-sequence builders (return input AND reference target)
# ============================================================
def make_probe_sequential(T=None, sq_center=50, tri_center=170, noise_std=0.0):
    T = T if T is not None else CFG.probe_T
    baseline = 0.05
    u = np.ones(T, dtype=np.float32) * baseline

    sq_start = sq_center - CFG.sq_width_max // 2
    u = np.maximum(u, make_square_pulse(T, sq_start, CFG.sq_width_max, CFG.sq_amp))

    tri_amp = (CFG.tri_amp_min + CFG.tri_amp_max) / 2
    u = add_triangle_pulse(u, tri_center, baseline, tri_amp, CFG.tri_rise, CFG.tri_fall)
    u += np.random.randn(T).astype(np.float32) * noise_std

    triggers = [
        (_trigger_of_square(sq_start), "square"),
        (_trigger_of_triangle(tri_center, baseline, tri_amp), "triangle"),
    ]
    ref = _build_reference(T, triggers)
    return u[:, None].astype(np.float32), ref


def make_probe_sq_tri_sq(T=None, sq1_center=40, tri_center=130, sq2_center=240,
                          noise_std=0.0):
    T = T if T is not None else CFG.probe_T
    baseline = 0.05
    u = np.ones(T, dtype=np.float32) * baseline

    sq1_start = sq1_center - CFG.sq_width_max // 2
    sq2_start = sq2_center - CFG.sq_width_max // 2
    u = np.maximum(u, make_square_pulse(T, sq1_start, CFG.sq_width_max, CFG.sq_amp))
    u = np.maximum(u, make_square_pulse(T, sq2_start, CFG.sq_width_max, CFG.sq_amp))

    tri_amp = (CFG.tri_amp_min + CFG.tri_amp_max) / 2
    u = add_triangle_pulse(u, tri_center, baseline, tri_amp, CFG.tri_rise, CFG.tri_fall)
    u += np.random.randn(T).astype(np.float32) * noise_std

    triggers = [
        (_trigger_of_square(sq1_start), "square"),
        (_trigger_of_triangle(tri_center, baseline, tri_amp), "triangle"),
        (_trigger_of_square(sq2_start), "square"),
    ]
    ref = _build_reference(T, triggers)
    return u[:, None].astype(np.float32), ref


def make_probe_intensity(T, shape, center, amp, noise_std=0.0):
    baseline = 0.05
    u = np.ones(T, dtype=np.float32) * baseline
    if shape == "square":
        sq_start = center - CFG.sq_width_max // 2
        u = np.maximum(u, make_square_pulse(T, sq_start, CFG.sq_width_max, amp))
    else:
        u = add_triangle_pulse(u, center, baseline, amp, CFG.tri_rise, CFG.tri_fall)
    u += np.random.randn(T).astype(np.float32) * noise_std

    triggers = []
    if amp > CFG.threshold:
        if shape == "square":
            triggers.append((_trigger_of_square(center - CFG.sq_width_max // 2), "square"))
        else:
            triggers.append((_trigger_of_triangle(center, baseline, amp), "triangle"))
    ref = _build_reference(T, triggers)
    return u[:, None].astype(np.float32), ref


# ============================================================
# 6. Model
# ============================================================
class LowRankRNN(nn.Module):
    """
    Low-rank RNN. Parameter count is IDENTICAL for the feedback and
    no-feedback variants: w_fb is always a trainable parameter. In the
    feedback variant it closes the loop from previous output. In the
    no-feedback variant it is repurposed as a second input projection.
    """

    def __init__(self, n_hidden=None, rank=None, input_dim=None, output_dim=None,
                 tau=None, dt=None, sigma_rec=None, use_feedback=True):
        super().__init__()
        self.n_hidden = n_hidden or CFG.n_hidden
        self.rank = rank or CFG.rank
        self.input_dim = input_dim or CFG.input_dim
        self.output_dim = output_dim or CFG.output_dim
        tau = tau or CFG.tau
        dt = dt or CFG.dt
        self.alpha = dt / tau
        self.sigma_rec = sigma_rec if sigma_rec is not None else CFG.sigma_rec
        self.use_feedback = use_feedback

        self.M = nn.Parameter(torch.randn(self.n_hidden, self.rank) / math.sqrt(self.n_hidden))
        self.N = nn.Parameter(torch.randn(self.n_hidden, self.rank) / math.sqrt(self.n_hidden))
        self.w_in = nn.Parameter(torch.randn(self.n_hidden, self.input_dim) / math.sqrt(self.input_dim))
        # Trainable in both modes so param counts match across variants.
        self.w_fb = nn.Parameter(torch.randn(self.n_hidden, self.output_dim) / math.sqrt(self.output_dim))
        self.w_out = nn.Parameter(torch.randn(self.output_dim, self.n_hidden) / math.sqrt(self.n_hidden))
        self.bias = nn.Parameter(torch.zeros(self.n_hidden))
        self.b_out = nn.Parameter(torch.zeros(self.output_dim))

    def recurrent_weight(self) -> torch.Tensor:
        return (self.M @ self.N.T) / self.n_hidden

    def forward(self, x, y_teacher=None, teacher_forcing_ratio=0.5,
                return_r=False, return_dynamics=False, override_use_feedback=None):
        B, T, _ = x.shape
        h = torch.zeros(B, self.n_hidden, device=x.device)
        r = torch.tanh(h)
        Wrec = self.recurrent_weight()
        y_prev = torch.zeros(B, self.output_dim, device=x.device)

        use_fb = self.use_feedback if override_use_feedback is None else override_use_feedback

        ys = []
        rs = [] if (return_r or return_dynamics) else None
        hs = [] if return_dynamics else None
        rec_terms = [] if return_dynamics else None
        inp_terms = [] if return_dynamics else None
        fb_terms = [] if return_dynamics else None

        for t in range(T):
            xt = x[:, t, :]

            if y_teacher is not None and t > 0:
                y_fb = y_teacher[:, t - 1, :] if torch.rand(1).item() < teacher_forcing_ratio else y_prev
            else:
                y_fb = y_prev

            rec_term = r @ Wrec.T
            inp_term = xt @ self.w_in.T
            # In no-feedback mode, w_fb becomes a second input projection.
            if use_fb:
                fb_term = y_fb @ self.w_fb.T
            else:
                fb_term = xt @ self.w_fb.T

            noise = self.sigma_rec * torch.randn_like(h)
            h = h + self.alpha * (-h + rec_term + inp_term + fb_term + self.bias + noise)
            r = torch.tanh(h)
            y_t = r @ self.w_out.T + self.b_out

            ys.append(y_t)
            if rs is not None:
                rs.append(r)
            if hs is not None:
                hs.append(h)
                rec_terms.append(rec_term)
                inp_terms.append(inp_term)
                fb_terms.append(fb_term)

            y_prev = y_t

        y_seq = torch.stack(ys, dim=1)
        if return_dynamics:
            return (y_seq, torch.stack(hs, dim=1), torch.stack(rs, dim=1),
                    torch.stack(rec_terms, dim=1), torch.stack(inp_terms, dim=1),
                    torch.stack(fb_terms, dim=1))
        if return_r:
            return y_seq, torch.stack(rs, dim=1)
        return y_seq

    def orthogonality_regularization(self) -> torch.Tensor:
        I = torch.eye(self.rank, device=self.M.device)
        return ((self.M.T @ self.M - I) ** 2).mean() + ((self.N.T @ self.N - I) ** 2).mean()

    @staticmethod
    def activity_regularization(r_seq: torch.Tensor) -> torch.Tensor:
        return (r_seq ** 2).mean()


# ============================================================
# 7. Training
# ============================================================
def compute_tf_ratio(epoch: int, total_epochs: int, use_feedback: bool) -> float:
    """
    Teacher-forcing schedule.

    Two phases:
      - Decay phase    : for the first (1 - tf_free_run_fraction) of epochs,
                         tf_ratio decays from tf_start down to tf_end using
                         the curve specified by CFG.tf_decay.
      - Free-run phase : for the final tf_free_run_fraction of epochs,
                         tf_ratio = 0 — the model drives its own feedback so
                         it must learn genuine post-stimulus dynamics.

    `epoch` is 1-indexed (as used in the training loop).
    """
    if not use_feedback:
        return 0.0
    free_run_start = max(1, int(total_epochs * (1.0 - CFG.tf_free_run_fraction)))
    if epoch > free_run_start:
        return 0.0
    progress = (epoch - 1) / max(1, free_run_start - 1)   # 0..1 across decay phase
    progress = min(1.0, max(0.0, progress))
    if CFG.tf_decay == "linear":
        factor = 1.0 - progress
    elif CFG.tf_decay == "cosine":
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    else:  # default: quadratic — decays fast early, lingers low
        factor = (1.0 - progress) ** 2
    return CFG.tf_end + (CFG.tf_start - CFG.tf_end) * factor


def train_model(train_ds, val_ds, use_feedback=True, save_name="model",
                save_dir=None, epochs=None, verbose=True) -> Dict:
    epochs = epochs or CFG.epochs
    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False, pin_memory=pin)

    model = LowRankRNN(use_feedback=use_feedback).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)

    train_losses, val_losses = [], []
    best_val = float("inf")
    best_state = None
    best_epoch = 0

    free_run_start = max(1, int(epochs * (1.0 - CFG.tf_free_run_fraction)))
    if verbose and use_feedback:
        print(f"[{save_name}] TF schedule: {CFG.tf_decay} decay "
              f"{CFG.tf_start}->{CFG.tf_end} over epochs 1..{free_run_start}, "
              f"then tf=0 for {epochs - free_run_start} free-run epochs.")

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        tf_ratio = compute_tf_ratio(epoch, epochs, use_feedback)

        run, n_items = 0.0, 0
        for x, y, w in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            w = w.to(device, non_blocking=True)
            pred, r_seq = model(
                x,
                y_teacher=y if use_feedback else None,
                teacher_forcing_ratio=tf_ratio,
                return_r=True,
            )
            # Weighted MSE: upweights the Type B tail region so the small-
            # amplitude damped wave contributes meaningfully to the gradient.
            weighted_sq = w * (pred - y) ** 2
            mse = weighted_sq.sum() / w.sum()
            loss = (mse
                    + 1e-3 * model.orthogonality_regularization()
                    + 1e-4 * model.activity_regularization(r_seq))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            run += loss.item() * x.size(0)
            n_items += x.size(0)
        train_losses.append(run / n_items)

        model.eval()
        run, n_items = 0.0, 0
        with torch.no_grad():
            # Validation uses plain (unweighted) MSE so the metric is
            # comparable across runs and weighting schemes.
            for x, y, _ in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                run += F.mse_loss(model(x), y).item() * x.size(0)
                n_items += x.size(0)
        val_loss = run / n_items
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            elapsed = time.time() - t0
            print(f"[{save_name}] Epoch {epoch:03d}/{epochs} | "
                  f"train={train_losses[-1]:.5f} | val={val_loss:.5f} | "
                  f"tf={tf_ratio:.3f} | best={best_val:.5f} | {elapsed:.1f}s")

    if save_dir is not None:
        ensure_dir(save_dir)
        torch.save(model.state_dict(), os.path.join(save_dir, f"{save_name}_last.pt"))
        torch.save(best_state, os.path.join(save_dir, f"{save_name}_best.pt"))

    best_model = LowRankRNN(use_feedback=use_feedback).to(device)
    best_model.load_state_dict(best_state)
    best_model.eval()
    model.eval()

    return {
        "last_model": model,
        "best_model": best_model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val": best_val,
        "best_epoch": best_epoch,
        "n_params": count_parameters(model),
    }


# ============================================================
# 8. Model evaluation helper
# ============================================================
def run_model_on_input(model, x_np, override_use_feedback=None):
    model.eval()
    x_t = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        y = model(x_t, override_use_feedback=override_use_feedback)
    return y[0].cpu().numpy()


# ============================================================
# 9. Plotting
# ============================================================
def plot_training_curves(curves: Dict[str, Dict[str, List[float]]], save_path=None):
    plt.figure(figsize=(8, 4))
    for label, d in curves.items():
        plt.plot(d["train"], label=f"Train {label}", alpha=0.8)
        plt.plot(d["val"], linestyle="--", label=f"Val {label}", alpha=0.8)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Training curves"); plt.legend(fontsize=8)
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_examples_by_type(model, dataset, n_per_type=3, save_dir=None, label=""):
    model.eval()
    if save_dir:
        ensure_dir(save_dir)
    for stype, stlabel in [(1, "typeA_square"), (2, "typeB_triangle")]:
        idxs = np.where(dataset.stim_types == stype)[0]
        if len(idxs) == 0:
            continue
        chosen = np.random.choice(idxs, size=min(n_per_type, len(idxs)), replace=False)
        for k, idx in enumerate(chosen):
            x, y, _ = dataset[int(idx)]
            with torch.no_grad():
                pred = model(x.unsqueeze(0).to(device))[0].cpu().numpy()
            x_np = x.numpy().squeeze(-1)
            y_np = y.numpy().squeeze(-1)
            pred_np = pred.squeeze(-1)
            fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
            axes[0].plot(x_np, color="steelblue")
            axes[0].axhline(CFG.threshold, linestyle=":", color="gray", label="Threshold")
            axes[0].set_ylabel("Input")
            axes[0].set_title(f"{label} — {stlabel} — example {k + 1}")
            axes[0].legend()
            axes[1].plot(y_np, label="Target", color="green")
            axes[1].plot(pred_np, "--", label="Predicted", color="darkorange")
            axes[1].set_ylabel("Output"); axes[1].set_xlabel("Time")
            axes[1].legend()
            plt.tight_layout()
            if save_dir:
                plt.savefig(os.path.join(save_dir, f"{stlabel}_example_{k}.png"),
                            dpi=200, bbox_inches="tight")
            plt.close()


def plot_model_outputs_grid(models, x_np, save_path, title="", reference=None):
    """Input + per-model output stacked in rows."""
    n = len(models)
    fig, axes = plt.subplots(n + 1, 1, figsize=(12, 1.6 * (n + 1)), sharex=True)
    axes[0].plot(x_np.squeeze(-1), color="steelblue")
    axes[0].axhline(CFG.threshold, linestyle=":", color="gray")
    axes[0].set_ylabel("Input")
    axes[0].set_title(title)

    for ax, (label, model) in zip(axes[1:], models.items()):
        pred = run_model_on_input(model, x_np)
        if reference is not None:
            ax.plot(reference.squeeze(-1), color="green", alpha=0.45, label="Reference")
        ax.plot(pred.squeeze(-1), color="darkorange", linewidth=1.5, label=label)
        ax.set_ylabel(label, fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_conditions_unified_grid(models, probes, save_path, title="", col_labels=None):
    """
    One big figure: rows = models, cols = test conditions.
    probes: list of (x_np, reference_np) tuples, one per column.
    col_labels: list of strings for column titles (same length as probes).
    Each cell overlays input (faded blue), reference (faded green), model output (orange).
    """
    n_rows = len(models)
    n_cols = len(probes)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.3 * n_cols, 1.6 * n_rows),
                             sharex=True, sharey=True)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[None, :]
    elif n_cols == 1:
        axes = axes[:, None]

    for c_idx, (x_np, ref) in enumerate(probes):
        for r_idx, (label, model) in enumerate(models.items()):
            pred = run_model_on_input(model, x_np)
            ax = axes[r_idx, c_idx]
            ax.plot(x_np.squeeze(-1), color="steelblue", alpha=0.35, linewidth=0.8)
            if ref is not None:
                ax.plot(ref.squeeze(-1), color="green", alpha=0.5, linewidth=0.8)
            ax.plot(pred.squeeze(-1), color="darkorange", linewidth=1.3)
            if r_idx == 0 and col_labels is not None:
                ax.set_title(col_labels[c_idx], fontsize=8)
            if c_idx == 0:
                ax.set_ylabel(label, fontsize=8)
    plt.suptitle(title, fontsize=11)
    plt.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_low_rank_projection(model, dataset, idx=0, save_path=None):
    model.eval()
    x, _, _ = dataset[idx]
    with torch.no_grad():
        _, _, r_seq, _, _, _ = model(x.unsqueeze(0).to(device), return_dynamics=True)
    r = r_seq[0].cpu().numpy()
    proj = r @ model.M.detach().cpu().numpy()
    plt.figure(figsize=(12, 4))
    if model.rank >= 2:
        plt.subplot(1, 2, 1)
        plt.plot(proj[:, 0], proj[:, 1], marker="o", markersize=2)
        plt.xlabel("Mode 1"); plt.ylabel("Mode 2")
        plt.title("Low-rank trajectory (modes 1-2)")
        plt.subplot(1, 2, 2)
        for k in range(min(4, model.rank)):
            plt.plot(proj[:, k], label=f"Mode {k+1}")
        plt.legend(); plt.title("Mode amplitudes over time")
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================
# 10. Tests on trained models
# ============================================================
def test_interval_sweep(models, intervals, save_dir):
    """
    interval = gap in timesteps between the end of the square pulse and the
    start of the triangle's rising phase.
    """
    ensure_dir(save_dir)
    metrics_cos = {label: [] for label in models}
    metrics_mse = {label: [] for label in models}
    sq_c = 40
    probes = []
    col_labels = []
    for intv in intervals:
        tri_c = sq_c + CFG.sq_width_max // 2 + intv + CFG.tri_rise
        x_np, ref = make_probe_sequential(T=CFG.probe_T, sq_center=sq_c, tri_center=tri_c)
        probes.append((x_np, ref))
        col_labels.append(f"interval={intv}")
        plot_model_outputs_grid(
            models, x_np,
            save_path=os.path.join(save_dir, f"interval_{intv:03d}.png"),
            title=f"Interval = {intv} timesteps",
            reference=ref,
        )
        for label, model in models.items():
            pred = run_model_on_input(model, x_np)
            m = compute_output_metrics(pred, ref)
            metrics_cos[label].append(m["cosine"])
            metrics_mse[label].append(m["mse"])

    # Unified grid: models × intervals
    plot_conditions_unified_grid(
        models, probes,
        save_path=os.path.join(save_dir, "unified_grid.png"),
        title="Interval sweep: all conditions",
        col_labels=col_labels,
    )

    plt.figure(figsize=(8, 4))
    for label, vals in metrics_cos.items():
        plt.plot(intervals, vals, marker="o", label=label)
    plt.xlabel("Interval (timesteps)"); plt.ylabel("Cosine similarity")
    plt.title("Interval sweep: output vs. reference")
    plt.ylim([-0.1, 1.05]); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "summary_cosine.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 4))
    for label, vals in metrics_mse.items():
        plt.plot(intervals, vals, marker="o", label=label)
    plt.xlabel("Interval (timesteps)"); plt.ylabel("MSE")
    plt.title("Interval sweep: MSE vs. reference")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "summary_mse.png"), dpi=200, bbox_inches="tight")
    plt.close()

    rows = []
    for label in models:
        for i, intv in enumerate(intervals):
            rows.append({"model": label, "interval": intv,
                         "cosine": metrics_cos[label][i], "mse": metrics_mse[label][i]})
    save_csv(rows, os.path.join(save_dir, "metrics.csv"))
    return metrics_cos


def test_sq_tri_sq(models, intervals, save_dir):
    """
    Sq → Tri → Sq with an interval sweep. `interval` is applied equally to
    both gaps: (end of sq1 → start of tri rise) and (end of tri fall → start
    of sq2). Matches the interval semantics used in test 1.
    """
    ensure_dir(save_dir)
    metrics_cos = {label: [] for label in models}
    metrics_mse = {label: [] for label in models}
    probes = []
    col_labels = []
    sq1_c = 40
    for intv in intervals:
        tri_c = sq1_c + CFG.sq_width_max // 2 + intv + CFG.tri_rise
        tri_fall_end = tri_c + CFG.tri_fall
        sq2_c = tri_fall_end + intv + CFG.sq_width_max // 2
        x_np, ref = make_probe_sq_tri_sq(T=CFG.probe_T, sq1_center=sq1_c,
                                          tri_center=tri_c, sq2_center=sq2_c)
        probes.append((x_np, ref))
        col_labels.append(f"interval={intv}")
        plot_model_outputs_grid(
            models, x_np,
            save_path=os.path.join(save_dir, f"sqtrisq_interval_{intv:03d}.png"),
            title=f"Sq → Tri → Sq, interval = {intv}",
            reference=ref,
        )
        for label, model in models.items():
            pred = run_model_on_input(model, x_np)
            m = compute_output_metrics(pred, ref)
            metrics_cos[label].append(m["cosine"])
            metrics_mse[label].append(m["mse"])

    plot_conditions_unified_grid(
        models, probes,
        save_path=os.path.join(save_dir, "unified_grid.png"),
        title="Sq → Tri → Sq: all intervals",
        col_labels=col_labels,
    )

    plt.figure(figsize=(8, 4))
    for label, vals in metrics_cos.items():
        plt.plot(intervals, vals, marker="o", label=label)
    plt.xlabel("Interval (timesteps)"); plt.ylabel("Cosine similarity")
    plt.title("Sq → Tri → Sq interval sweep: cosine vs reference")
    plt.ylim([-0.1, 1.05]); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "summary_cosine.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 4))
    for label, vals in metrics_mse.items():
        plt.plot(intervals, vals, marker="o", label=label)
    plt.xlabel("Interval (timesteps)"); plt.ylabel("MSE")
    plt.title("Sq → Tri → Sq interval sweep: MSE vs reference")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "summary_mse.png"), dpi=200, bbox_inches="tight")
    plt.close()

    rows = []
    for label in models:
        for i, intv in enumerate(intervals):
            rows.append({"model": label, "interval": intv,
                         "cosine": metrics_cos[label][i], "mse": metrics_mse[label][i]})
    save_csv(rows, os.path.join(save_dir, "metrics.csv"))
    return metrics_cos


def test_intensity_sweep(models, save_dir, shape="triangle"):
    ensure_dir(save_dir)
    if shape == "triangle":
        amps = np.linspace(0.3, 2.0, 20)
    else:
        amps = np.linspace(0.3, 2.0, 20)
    T = CFG.probe_T
    center = 100

    cos_by = {label: [] for label in models}
    peak_by = {label: [] for label in models}

    for amp in amps:
        x_np, ref = make_probe_intensity(T=T, shape=shape, center=center, amp=float(amp))
        for label, model in models.items():
            pred = run_model_on_input(model, x_np)
            m = compute_output_metrics(pred, ref)
            cos_by[label].append(m["cosine"])
            peak_by[label].append(float(pred.max()))

    # Grid plot: rows = models, cols = 6 subsampled amps
    sub_idx = np.linspace(0, len(amps) - 1, 6).astype(int)
    n_models = len(models)
    fig, axes = plt.subplots(n_models, len(sub_idx),
                             figsize=(2.5 * len(sub_idx), 1.7 * n_models),
                             sharex=True, sharey=True)
    if n_models == 1:
        axes = axes[None, :]
    for r_idx, (label, model) in enumerate(models.items()):
        for c_idx, i in enumerate(sub_idx):
            amp = float(amps[i])
            x_np, ref = make_probe_intensity(T=T, shape=shape, center=center, amp=amp)
            pred = run_model_on_input(model, x_np)
            axes[r_idx, c_idx].plot(x_np.squeeze(-1), color="steelblue", alpha=0.5, linewidth=0.8)
            axes[r_idx, c_idx].plot(ref.squeeze(-1), color="green", alpha=0.5, linewidth=0.8)
            axes[r_idx, c_idx].plot(pred.squeeze(-1), color="darkorange", linewidth=1.3)
            if r_idx == 0:
                axes[r_idx, c_idx].set_title(f"amp={amp:.2f}", fontsize=8)
            if c_idx == 0:
                axes[r_idx, c_idx].set_ylabel(label, fontsize=8)
    plt.suptitle(f"Intensity sweep ({shape})", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"grid_{shape}.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 4))
    for label in models:
        plt.plot(amps, cos_by[label], marker="o", label=label)
    plt.axvline(CFG.threshold, linestyle=":", color="gray", label="Threshold")
    plt.xlabel(f"{shape} amplitude"); plt.ylabel("Cosine similarity")
    plt.title(f"Intensity sweep: cosine ({shape})")
    plt.ylim([-0.2, 1.05]); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"cosine_{shape}.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 4))
    for label in models:
        plt.plot(amps, peak_by[label], marker="o", label=label)
    plt.axvline(CFG.threshold, linestyle=":", color="gray", label="Threshold")
    plt.xlabel(f"{shape} amplitude"); plt.ylabel("Output peak")
    plt.title(f"Intensity sweep: output peak ({shape})")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"peak_{shape}.png"), dpi=200, bbox_inches="tight")
    plt.close()

    rows = []
    for label in models:
        for i, amp in enumerate(amps):
            rows.append({"model": label, "shape": shape, "amp": float(amp),
                         "cosine": cos_by[label][i], "peak": peak_by[label][i]})
    save_csv(rows, os.path.join(save_dir, f"metrics_{shape}.csv"))

    return list(amps), cos_by, peak_by


def plot_similarity_summary(all_metrics, save_path):
    """all_metrics: {probe_name: {model_label: cosine_value}}."""
    probes = list(all_metrics.keys())
    labels = list(next(iter(all_metrics.values())).keys())
    n_probes = len(probes)
    n_models = len(labels)
    x = np.arange(n_probes)
    width = 0.8 / max(1, n_models)
    plt.figure(figsize=(10, 5))
    for i, label in enumerate(labels):
        vals = [all_metrics[p][label] for p in probes]
        plt.bar(x + (i - n_models / 2) * width + width / 2, vals, width=width, label=label)
    plt.xticks(x, probes, rotation=20, ha="right")
    plt.ylabel("Cosine similarity")
    plt.title("Probe cosine similarity across models")
    plt.ylim([-0.1, 1.05]); plt.legend(fontsize=8); plt.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================
# 11. Main: training + probes
# ============================================================
def run_single_training(base_dir: str):
    probe_dir = os.path.join(base_dir, "probes")
    fb_dir = os.path.join(base_dir, "with_feedback")
    nofb_dir = os.path.join(base_dir, "no_feedback")
    compare_dir = os.path.join(base_dir, "compare")
    for d in [base_dir, probe_dir, fb_dir, nofb_dir, compare_dir]:
        ensure_dir(d)

    save_csv([asdict(CFG)], os.path.join(base_dir, "config.csv"))

    print("Generating datasets...")
    set_seed(DEFAULT_SEED)
    train_ds = TwoStimDataset(n_samples=CFG.train_samples, T=CFG.train_T,
                               threshold=CFG.threshold)
    val_ds = TwoStimDataset(n_samples=CFG.val_samples, T=CFG.train_T,
                             threshold=CFG.threshold)
    typeA = int((train_ds.stim_types == 1).sum())
    typeB = int((train_ds.stim_types == 2).sum())
    neg = int((train_ds.stim_types == 0).sum())
    print(f"Train dataset: TypeA={typeA}, TypeB={typeB}, Negative={neg}")

    print("\nTraining WITH feedback...")
    fb = train_model(train_ds, val_ds, use_feedback=True,
                     save_name="with_feedback", save_dir=fb_dir)
    print(f"  FB n_params={fb['n_params']} | best@epoch {fb['best_epoch']}={fb['best_val']:.5f}")

    print("\nTraining WITHOUT feedback...")
    nofb = train_model(train_ds, val_ds, use_feedback=False,
                       save_name="no_feedback", save_dir=nofb_dir)
    print(f"  noFB n_params={nofb['n_params']} | best@epoch {nofb['best_epoch']}={nofb['best_val']:.5f}")

    assert fb["n_params"] == nofb["n_params"], \
        f"Param mismatch: FB={fb['n_params']}, noFB={nofb['n_params']}"
    print(f"  Parameter count matched: {fb['n_params']}")

    plot_training_curves({
        "FB": {"train": fb["train_losses"], "val": fb["val_losses"]},
        "noFB": {"train": nofb["train_losses"], "val": nofb["val_losses"]},
    }, save_path=os.path.join(compare_dir, "training_curves.png"))

    models = {
        "FB-last": fb["last_model"],
        "FB-best": fb["best_model"],
        "noFB-last": nofb["last_model"],
        "noFB-best": nofb["best_model"],
    }

    # Training-set examples per model
    print("\nPlotting per-model training-set examples...")
    for label, m in models.items():
        plot_examples_by_type(m, val_ds, n_per_type=3,
                               save_dir=os.path.join(base_dir, "examples", label),
                               label=label)

    # Baseline sequential probe (for reference)
    print("Baseline sequential probe...")
    x_base, ref_base = make_probe_sequential(T=CFG.probe_T, sq_center=50, tri_center=170)
    plot_model_outputs_grid(models, x_base,
                             save_path=os.path.join(probe_dir, "baseline_sequential.png"),
                             title="Baseline: small square → large triangle",
                             reference=ref_base)

    # Test 1: Interval sweep
    print("\n[Test 1] Interval sweep [0, 2, 4, 7, 10, 15, 20]...")
    intervals = [-1, 0, 2, 4, 7, 10, 15, 20, 40]
    cos_by_interval = test_interval_sweep(
        models, intervals,
        save_dir=os.path.join(probe_dir, "test1_interval"),
    )

    # Test 2: Sq → Tri → Sq (interval sweep, same intervals as test 1)
    print("[Test 2] Sq → Tri → Sq interval sweep [0, 2, 4, 7, 10, 15, 20]...")
    sqtrisq_intervals = [0, 5, 10, 20, 40]
    cos_by_sqtrisq = test_sq_tri_sq(
        models, sqtrisq_intervals,
        save_dir=os.path.join(probe_dir, "test2_sq_tri_sq"),
    )

    # Test 3: Intensity sweep
    print("[Test 3] Intensity sweep (triangle)...")
    amps_tri, cos_tri, _ = test_intensity_sweep(
        models, save_dir=os.path.join(probe_dir, "test3_intensity"), shape="triangle",
    )
    print("[Test 3] Intensity sweep (square)...")
    amps_sq, cos_sq, _ = test_intensity_sweep(
        models, save_dir=os.path.join(probe_dir, "test3_intensity"), shape="square",
    )

    # Cross-probe summary
    summary = {
        "baseline": {label: compute_output_metrics(
            run_model_on_input(m, x_base), ref_base)["cosine"] for label, m in models.items()},
        "interval=0": {label: cos_by_interval[label][intervals.index(0)] for label in models},
        "interval=10": {label: cos_by_interval[label][intervals.index(10)] for label in models},
        "interval=20": {label: cos_by_interval[label][intervals.index(20)] for label in models},
        **{f"sq_tri_sq_int={sqtrisq_intervals[i]}":
           {label: cos_by_sqtrisq[label][i] for label in models}
           for i in (0, len(sqtrisq_intervals) // 2, len(sqtrisq_intervals) - 1)},
        "tri_amp=1.0": {label: cos_tri[label][int(np.argmin(np.abs(np.array(amps_tri) - 1.0)))]
                        for label in models},
        "sq_amp=0.7": {label: cos_sq[label][int(np.argmin(np.abs(np.array(amps_sq) - 0.7)))]
                       for label in models},
    }
    plot_similarity_summary(summary, save_path=os.path.join(compare_dir, "cosine_summary.png"))
    rows = [{"probe": p, "model": m, "cosine": v}
            for p, d in summary.items() for m, v in d.items()]
    save_csv(rows, os.path.join(compare_dir, "cosine_summary.csv"))

    # Low-rank projections
    for label, m in models.items():
        plot_low_rank_projection(m, val_ds, idx=0,
                                  save_path=os.path.join(base_dir, "lowrank", f"{label}.png"))

    return models, fb, nofb


def run_multi_seed(base_dir: str):
    multi_dir = os.path.join(base_dir, "multi_seed")
    ensure_dir(multi_dir)

    results_fb, results_nofb = [], []
    for seed in CFG.multi_seeds:
        print(f"\n[Multi-seed] seed={seed} FB")
        set_seed(seed)
        t_ds = TwoStimDataset(n_samples=CFG.multi_seed_samples, T=CFG.train_T,
                               threshold=CFG.threshold)
        v_ds = TwoStimDataset(n_samples=CFG.val_samples, T=CFG.train_T,
                               threshold=CFG.threshold)
        r = train_model(t_ds, v_ds, use_feedback=True,
                        save_name=f"fb_seed{seed}",
                        save_dir=os.path.join(multi_dir, f"fb_seed{seed}"),
                        epochs=CFG.multi_seed_epochs, verbose=False)
        r["seed"] = seed
        results_fb.append(r)
        print(f"  best val = {r['best_val']:.5f}")

        print(f"[Multi-seed] seed={seed} noFB")
        set_seed(seed)
        t_ds = TwoStimDataset(n_samples=CFG.multi_seed_samples, T=CFG.train_T,
                               threshold=CFG.threshold)
        v_ds = TwoStimDataset(n_samples=CFG.val_samples, T=CFG.train_T,
                               threshold=CFG.threshold)
        r = train_model(t_ds, v_ds, use_feedback=False,
                        save_name=f"nofb_seed{seed}",
                        save_dir=os.path.join(multi_dir, f"nofb_seed{seed}"),
                        epochs=CFG.multi_seed_epochs, verbose=False)
        r["seed"] = seed
        results_nofb.append(r)
        print(f"  best val = {r['best_val']:.5f}")

    # Training curves across seeds
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for r in results_fb:
        axes[0].plot(r["val_losses"], alpha=0.7, label=f"FB seed={r['seed']}")
    for r in results_nofb:
        axes[0].plot(r["val_losses"], alpha=0.7, linestyle="--", label=f"noFB seed={r['seed']}")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Val loss")
    axes[0].set_title("Validation curves across seeds"); axes[0].legend(fontsize=7)

    labels = [f"FB-{r['seed']}" for r in results_fb] + [f"noFB-{r['seed']}" for r in results_nofb]
    best_vals = [r["best_val"] for r in results_fb] + [r["best_val"] for r in results_nofb]
    axes[1].bar(labels, best_vals)
    axes[1].set_ylabel("Best val loss")
    axes[1].set_title("Best val loss per seed")
    for tick in axes[1].get_xticklabels():
        tick.set_rotation(30); tick.set_horizontalalignment("right")
    plt.tight_layout()
    plt.savefig(os.path.join(multi_dir, "training_curves.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Baseline probe across seeds
    x_np, ref = make_probe_sequential(T=CFG.probe_T, sq_center=50, tri_center=170)
    models = {}
    for r in results_fb:
        models[f"FB-s{r['seed']}"] = r["best_model"]
    for r in results_nofb:
        models[f"noFB-s{r['seed']}"] = r["best_model"]
    plot_model_outputs_grid(models, x_np,
                             save_path=os.path.join(multi_dir, "baseline_probe_overlay.png"),
                             title="Baseline probe — all seeds", reference=ref)

    # Cosine per seed
    rows = []
    for label, m in models.items():
        pred = run_model_on_input(m, x_np)
        rows.append({"model": label, **compute_output_metrics(pred, ref)})
    save_csv(rows, os.path.join(multi_dir, "seed_metrics.csv"))

    plt.figure(figsize=(9, 4))
    plt.bar([r["model"] for r in rows], [r["cosine"] for r in rows])
    plt.ylim([-0.1, 1.05]); plt.ylabel("Cosine similarity (baseline)")
    plt.title("Seed variability: baseline probe cosine")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(multi_dir, "seed_cosine_bar.png"), dpi=200, bbox_inches="tight")
    plt.close()

    return results_fb, results_nofb


if __name__ == "__main__":
    print(f"Device: {device}")
    base_dir = CFG.base_dir
    models, fb, nofb = run_single_training(base_dir)
    run_multi_seed(base_dir)
    print(f"\nAll results saved to: {base_dir}")
