"""
Analysis of trained LowRankRNN checkpoints.

Runs three families of tests to understand:
  1) what feedback *is* (weight structure)
  2) what feedback *does* (causal ablation)
  3) what feedback *enables* (state-space / dynamics)

Checkpoints are expected at:
  results/with_feedback/with_feedback_{last,best}.pt
  results/no_feedback/no_feedback_{last,best}.pt

Outputs land under results/analysis/{A,B,C}/.
"""

from __future__ import annotations

import argparse
import copy
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import Ridge

from main import (
    CFG,
    LowRankRNN,
    _build_reference,
    _trigger_of_triangle,
    compute_output_metrics,
    device,
    ensure_dir,
    make_probe_sequential,
    make_probe_intensity,
    make_probe_sq_tri_sq,
    save_csv,
    set_seed,
)


# ============================================================
# Config
# ============================================================
CHECKPOINTS = {
    "FB-best": ("with_feedback/with_feedback_best.pt", True),
    "FB-last": ("with_feedback/with_feedback_last.pt", True),
    "noFB-best": ("no_feedback/no_feedback_best.pt", False),
    "noFB-last": ("no_feedback/no_feedback_last.pt", False),
}

# Models to run ablations on. Ablations are interesting for feedback
# models; for noFB they test perturbations of the repurposed input path.
ABLATION_MODELS = ["FB-best", "noFB-best"]

ANALYSIS_SEED = 7


# ============================================================
# Loading / basic utilities
# ============================================================
def load_models(base_dir: str) -> Dict[str, LowRankRNN]:
    """
    Load each checkpoint, inferring geometry (n_hidden, rank, input_dim,
    output_dim) from the stored tensors rather than from CFG — CFG may have
    drifted since training.
    """
    models: Dict[str, LowRankRNN] = {}
    for label, (rel, use_fb) in CHECKPOINTS.items():
        ckpt = os.path.join(base_dir, rel)
        if not os.path.isfile(ckpt):
            print(f"[load_models] skipping missing checkpoint: {ckpt}")
            continue
        state = torch.load(ckpt, map_location=device)
        n_hidden, rank = state["M"].shape
        input_dim = state["w_in"].shape[1]
        output_dim = state["w_fb"].shape[1]
        m = LowRankRNN(n_hidden=n_hidden, rank=rank,
                       input_dim=input_dim, output_dim=output_dim,
                       use_feedback=use_fb).to(device)
        m.load_state_dict(state)
        m.eval()
        models[label] = m
        print(f"[load_models] {label}: n_hidden={n_hidden}, rank={rank}, "
              f"input={input_dim}, output={output_dim}")
    if not models:
        raise FileNotFoundError(f"No checkpoints found under {base_dir}")
    return models


def square_probe(sq_center: int = 80, T: Optional[int] = None):
    """
    Square-only probe. Reference is the small output triangle triggered
    by the square. Windows: pre / main / post.
    """
    from main import make_probe_intensity
    T = T if T is not None else CFG.probe_T
    x_np, ref = make_probe_intensity(T=T, shape="square",
                                      center=sq_center, amp=CFG.sq_amp)
    sq_start = sq_center - CFG.sq_width_max // 2
    main_start = sq_start
    main_end = sq_start + CFG.small_out_rise + CFG.small_out_fall
    windows = {
        "pre": (max(0, sq_start - 20), sq_start),
        "main": (main_start, main_end),
        "post": (main_end, min(T, main_end + 60)),
        "all": (0, T),
    }
    return x_np, ref, windows


def triangle_probe(tri_amp: float = 1.175, tri_center: int = 170,
                   T: Optional[int] = None):
    """
    Build a triangle-only probe (no preceding square) whose reference has
    the large triangle plus the damped-wave tail.

    Returns:
      x_np:    (T, 1) input
      ref_np:  (T, 1) reference target
      windows: dict of {name: (t0, t1)} demarcating main bump vs tail.
    """
    T = T if T is not None else CFG.probe_T
    from main import add_triangle_pulse  # local import to avoid reshuffling top-level

    baseline = 0.05
    u = np.ones(T, dtype=np.float32) * baseline
    u = add_triangle_pulse(u, tri_center, baseline, tri_amp,
                           CFG.tri_rise, CFG.tri_fall)

    trigger = _trigger_of_triangle(tri_center, baseline, tri_amp)
    ref = _build_reference(T, [(trigger, "triangle")])

    main_start = trigger
    main_end = trigger + CFG.large_out_rise + CFG.large_out_fall
    tail_start = main_end
    tail_end = min(T, main_end + CFG.wave_duration)
    windows = {
        "pre": (max(0, trigger - 20), trigger),
        "main": (main_start, main_end),
        "tail": (tail_start, tail_end),
        "all": (0, T),
    }
    return u[:, None].astype(np.float32), ref, windows


def windowed_metrics(pred: np.ndarray, ref: np.ndarray,
                     windows: Dict[str, Tuple[int, int]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for name, (t0, t1) in windows.items():
        p = pred[t0:t1]
        r = ref[t0:t1]
        if p.size == 0 or r.size == 0:
            out[name] = {"cosine": 0.0, "mse": 0.0}
            continue
        out[name] = compute_output_metrics(p, r)
    return out


# ============================================================
# Custom forward with modifiers (powers Section B)
# ============================================================
@torch.no_grad()
def forward_modified(
    model: LowRankRNN,
    x_np: np.ndarray,
    fb_gain: float = 1.0,
    fb_delay: int = 0,
    fb_lesion_window: Optional[Tuple[int, int]] = None,
    fb_noise_std: float = 0.0,
    fb_freeze_after: Optional[int] = None,
    fb_freeze_value: Optional[float] = None,
    input_noise_std: float = 0.0,
    rec_noise_std: Optional[float] = None,
    return_hidden: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Re-implements the model's forward loop so we can inject knobs:
      fb_gain              scalar multiplier on fb_term
      fb_delay             use y_prev from delta steps ago
      fb_lesion_window     (t0, t1) during which fb_term is zeroed
      fb_noise_std         additive Gaussian noise on fb_term
      fb_freeze_after      after this step, replace fb_term with freeze_value
      input_noise_std      additive Gaussian noise on x_t
      rec_noise_std        overrides model.sigma_rec for this run
    Works for both FB and noFB variants (for noFB, "fb" is the repurposed
    second input pathway).
    """
    T = x_np.shape[0]
    x = torch.from_numpy(x_np).float().to(device).unsqueeze(0)  # (1, T, 1)
    sigma_rec = model.sigma_rec if rec_noise_std is None else rec_noise_std

    h = torch.zeros(1, model.n_hidden, device=device)
    r = torch.tanh(h)
    Wrec = model.recurrent_weight()
    y_prev = torch.zeros(1, model.output_dim, device=device)

    # ring buffer of past outputs, indexed by delay
    buf = [torch.zeros(1, model.output_dim, device=device)] * max(1, fb_delay + 1)

    ys, rs = [], []
    for t in range(T):
        xt = x[:, t, :]
        if input_noise_std > 0:
            xt = xt + input_noise_std * torch.randn_like(xt)

        if model.use_feedback:
            y_feed = buf[-(fb_delay + 1)] if fb_delay > 0 else y_prev
            fb_term = y_feed @ model.w_fb.T
        else:
            fb_term = xt @ model.w_fb.T  # repurposed input path

        fb_term = fb_term * fb_gain
        if fb_noise_std > 0:
            fb_term = fb_term + fb_noise_std * torch.randn_like(fb_term)
        if fb_lesion_window is not None:
            t0, t1 = fb_lesion_window
            if t0 <= t < t1:
                fb_term = torch.zeros_like(fb_term)
        if fb_freeze_after is not None and t >= fb_freeze_after:
            val = 0.0 if fb_freeze_value is None else float(fb_freeze_value)
            fb_term = torch.full_like(fb_term, val)

        rec_term = r @ Wrec.T
        inp_term = xt @ model.w_in.T
        noise = sigma_rec * torch.randn_like(h)
        h = h + model.alpha * (-h + rec_term + inp_term + fb_term + model.bias + noise)
        r = torch.tanh(h)
        y_t = r @ model.w_out.T + model.b_out

        ys.append(y_t)
        rs.append(r)

        y_prev = y_t
        if fb_delay > 0:
            buf.append(y_t)
            buf = buf[-(fb_delay + 1):]

    y_seq = torch.stack(ys, dim=1).squeeze(0).cpu().numpy()
    r_seq = torch.stack(rs, dim=1).squeeze(0).cpu().numpy() if return_hidden else None
    return y_seq, r_seq


@torch.no_grad()
def run_model(model: LowRankRNN, x_np: np.ndarray) -> np.ndarray:
    """Plain inference — no ablation."""
    y, _ = forward_modified(model, x_np)
    return y


# ============================================================
# Metrics on vectors / weights
# ============================================================
def gini(x: np.ndarray) -> float:
    x = np.abs(np.asarray(x, dtype=np.float64)).flatten()
    if x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    return float((n + 1 - 2 * cum.sum() / cum[-1]) / n) * -1.0 + 1.0 - 1.0 / n


def participation_ratio(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).flatten()
    s2 = (x ** 2)
    denom = (s2 ** 2).sum()
    if denom == 0:
        return 0.0
    return float(s2.sum() ** 2 / denom)


def unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + eps)


# ============================================================
# SECTION A: weight-level analysis
# ============================================================
def section_A(models: Dict[str, LowRankRNN], save_dir: str) -> None:
    ensure_dir(save_dir)
    A1_feedback_distribution(models, save_dir)
    A2_alignment(models, save_dir)
    A4_mode_participation(models, save_dir)
    A5_neuron_classification(models, save_dir)


def A1_feedback_distribution(models, save_dir):
    """Histogram of |w_fb|; Gini and participation ratio per model."""
    out_dir = os.path.join(save_dir, "A1_fb_distribution")
    ensure_dir(out_dir)

    rows = []
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(3.5 * n_models, 3.2), sharey=True)
    if n_models == 1:
        axes = [axes]
    for ax, (label, m) in zip(axes, models.items()):
        wfb = m.w_fb.detach().cpu().numpy().flatten()
        absv = np.abs(wfb)
        ax.hist(absv, bins=40, color="steelblue", alpha=0.8)
        ax.set_title(f"{label}\n|w_fb| distribution", fontsize=9)
        ax.set_xlabel("|w_fb[i]|")
        ax.set_ylabel("# neurons")
        rows.append({
            "model": label,
            "n_hidden": int(absv.size),
            "mean_abs": float(absv.mean()),
            "max_abs": float(absv.max()),
            "std": float(absv.std()),
            "gini": gini(absv),
            "participation_ratio": participation_ratio(absv),
            "pr_fraction": participation_ratio(absv) / absv.size,
        })
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "histograms.png"), dpi=200, bbox_inches="tight")
    plt.close()
    save_csv(rows, os.path.join(out_dir, "metrics.csv"))

    # bar chart of Gini and PR across models
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    labels = [r["model"] for r in rows]
    axes[0].bar(labels, [r["gini"] for r in rows], color="teal")
    axes[0].set_title("Gini(|w_fb|)  —  higher = more concentrated")
    axes[0].set_ylim(0, 1)
    axes[1].bar(labels, [r["pr_fraction"] for r in rows], color="darkorange")
    axes[1].set_title("PR(|w_fb|) / n_hidden  —  1 = uniform")
    axes[1].set_ylim(0, 1)
    for ax in axes:
        for t in ax.get_xticklabels():
            t.set_rotation(20); t.set_horizontalalignment("right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "concentration_metrics.png"), dpi=200, bbox_inches="tight")
    plt.close()


def A2_alignment(models, save_dir):
    """Cosine of w_fb with w_out and with each column of M."""
    out_dir = os.path.join(save_dir, "A2_alignment")
    ensure_dir(out_dir)

    rows = []
    for label, m in models.items():
        wfb = m.w_fb.detach().cpu().numpy().flatten()
        wout = m.w_out.detach().cpu().numpy().flatten()
        Mm = m.M.detach().cpu().numpy()        # (n_hidden, rank)
        Nn = m.N.detach().cpu().numpy()        # (n_hidden, rank)

        cos_fb_out = float(np.dot(unit(wfb), unit(wout)))
        modes = {}
        for k in range(Mm.shape[1]):
            modes[f"cos_fb_M{k}"] = float(np.dot(unit(wfb), unit(Mm[:, k])))
            modes[f"cos_fb_N{k}"] = float(np.dot(unit(wfb), unit(Nn[:, k])))
            modes[f"cos_out_M{k}"] = float(np.dot(unit(wout), unit(Mm[:, k])))
        rows.append({"model": label, "cos_fb_out": cos_fb_out, **modes})

    save_csv(rows, os.path.join(out_dir, "alignment.csv"))

    # Heatmap-style bar plot
    keys = [k for k in rows[0].keys() if k != "model"]
    labels = [r["model"] for r in rows]
    vals = np.array([[r[k] for k in keys] for r in rows])  # (n_models, n_keys)

    fig, ax = plt.subplots(figsize=(1.0 * len(keys) + 2, 0.5 * len(labels) + 2))
    im = ax.imshow(vals, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(keys)):
            ax.text(j, i, f"{vals[i, j]:+.2f}", ha="center", va="center",
                    fontsize=7, color="black")
    plt.colorbar(im, ax=ax, fraction=0.04)
    plt.title("Weight-vector alignments (cosine)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "alignment_heatmap.png"), dpi=200, bbox_inches="tight")
    plt.close()


def A4_mode_participation(models, save_dir):
    """
    Project w_fb onto the orthonormalized basis of M (the 'readout-mode'
    subspace). Report energy per mode vs energy orthogonal to it.
    Also compare how w_fb vs w_in vs N distribute energy across those modes.
    """
    out_dir = os.path.join(save_dir, "A4_mode_participation")
    ensure_dir(out_dir)

    rows = []
    for label, m in models.items():
        Mm = m.M.detach().cpu().numpy()
        Q, _ = np.linalg.qr(Mm)     # orthonormal basis of col(M), shape (n_hidden, rank)
        wfb = m.w_fb.detach().cpu().numpy().flatten()
        win = m.w_in.detach().cpu().numpy().flatten()
        wout = m.w_out.detach().cpu().numpy().flatten()
        Nn = m.N.detach().cpu().numpy()

        def proj_energy(v):
            v = v.flatten()
            coeffs = Q.T @ v                 # shape (rank,)
            in_sub = float((coeffs ** 2).sum())
            total = float((v ** 2).sum())
            ortho = max(0.0, total - in_sub)
            return coeffs, in_sub, ortho, total

        fb_c, fb_in, fb_o, fb_tot = proj_energy(wfb)
        in_c, in_in, in_o, in_tot = proj_energy(win)
        out_c, out_in, out_o, out_tot = proj_energy(wout)

        row = {"model": label,
               "fb_frac_in_M": fb_in / (fb_tot + 1e-12),
               "in_frac_in_M": in_in / (in_tot + 1e-12),
               "out_frac_in_M": out_in / (out_tot + 1e-12)}
        for k in range(Mm.shape[1]):
            row[f"fb_coef_M{k}"] = float(fb_c[k])
            row[f"in_coef_M{k}"] = float(in_c[k])
            row[f"out_coef_M{k}"] = float(out_c[k])
            # N already spans its own rank-r subspace; compare to M
            row[f"cos_Nk_M{k}"] = float(np.dot(unit(Nn[:, k]), unit(Mm[:, k])))
        rows.append(row)

    save_csv(rows, os.path.join(out_dir, "mode_participation.csv"))

    # Plot: per-model bar of energy fraction inside M subspace.
    labels = [r["model"] for r in rows]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(x - width, [r["fb_frac_in_M"] for r in rows], width, label="w_fb")
    ax.bar(x,         [r["in_frac_in_M"] for r in rows], width, label="w_in")
    ax.bar(x + width, [r["out_frac_in_M"] for r in rows], width, label="w_out")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Fraction of energy inside col(M)")
    ax.set_title("How much of each weight vector lives in the readout-mode subspace")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "energy_in_M_subspace.png"), dpi=200, bbox_inches="tight")
    plt.close()


def _spearman_r(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def A5_neuron_classification(models, save_dir):
    """
    Expanded A5. Main targets:
      (i)  Do high-|w_fb| neurons systematically have higher |w_in| or
           higher |w_out|?  Shown as stratified means (top-K vs rest),
           Spearman rank correlations, and quadrant counts.
      (ii) How do high-|w_fb| neurons behave under the *square* stimulus?
      (iii) Peak-time analysis: during the triangle tail, do the top-K
           feedback-heavy neurons peak at different times? If yes, that's
           a temporal-basis / phase-code signature.
    """
    out_dir = os.path.join(save_dir, "A5_neuron_classification")
    ensure_dir(out_dir)
    x_tri, ref_tri, win_tri = triangle_probe()
    x_sq, ref_sq, win_sq = square_probe()

    for label, m in models.items():
        fb_s = np.abs(m.w_fb.detach().cpu().numpy()).flatten()
        in_s = np.abs(m.w_in.detach().cpu().numpy()).flatten()
        out_s = np.abs(m.w_out.detach().cpu().numpy()).flatten()
        n = fb_s.size

        # ------------------------------------------------------------
        # (i) fb vs in/out: stratified analysis
        # ------------------------------------------------------------
        K = max(4, n // 8)  # top-K feedback-heavy neurons (e.g. 16 of 128)
        top_idx = np.argsort(fb_s)[::-1][:K]
        rest_idx = np.argsort(fb_s)[::-1][K:]

        strat_rows = []
        for name, idx in [("top-K fb", top_idx), ("rest", rest_idx), ("all", np.arange(n))]:
            strat_rows.append({
                "model": label, "group": name, "n": int(len(idx)),
                "mean_|w_fb|": float(fb_s[idx].mean()),
                "mean_|w_in|": float(in_s[idx].mean()),
                "mean_|w_out|": float(out_s[idx].mean()),
            })
        save_csv(strat_rows, os.path.join(out_dir, f"stratified_{label}.csv"))

        # Pearson + Spearman (rank) correlations — the rank version is
        # robust against a few very large weights dominating the fit.
        corrs = {
            "pearson_fb_in":  float(np.corrcoef(fb_s, in_s)[0, 1]),
            "pearson_fb_out": float(np.corrcoef(fb_s, out_s)[0, 1]),
            "spearman_fb_in":  _spearman_r(fb_s, in_s),
            "spearman_fb_out": _spearman_r(fb_s, out_s),
        }
        save_csv([{"model": label, **corrs}], os.path.join(out_dir, f"corrs_{label}.csv"))

        # Quadrant counts: median-split on fb, then median-split on in/out,
        # so we can see which quadrant the fb-heavy neurons fall into.
        fb_hi = fb_s > np.median(fb_s)
        in_hi = in_s > np.median(in_s)
        out_hi = out_s > np.median(out_s)
        quad_in = np.array([
            [np.sum(~fb_hi & ~in_hi), np.sum(~fb_hi & in_hi)],
            [np.sum(fb_hi & ~in_hi),  np.sum(fb_hi & in_hi)],
        ])
        quad_out = np.array([
            [np.sum(~fb_hi & ~out_hi), np.sum(~fb_hi & out_hi)],
            [np.sum(fb_hi & ~out_hi),  np.sum(fb_hi & out_hi)],
        ])

        # Visual: stratified means + scatter with top-K highlighted
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        # Row 0: bar chart of stratified means
        groups = ["top-K fb", "rest"]
        x0 = np.arange(len(groups))
        width = 0.35
        axes[0, 0].bar(x0 - width / 2,
                       [fb_s[top_idx].mean(), fb_s[rest_idx].mean()],
                       width, label="|w_fb|", color="crimson")
        axes[0, 0].set_xticks(x0); axes[0, 0].set_xticklabels(groups)
        axes[0, 0].set_title("|w_fb| by group"); axes[0, 0].legend(fontsize=8)

        axes[0, 1].bar(x0 - width / 2,
                       [in_s[top_idx].mean(), in_s[rest_idx].mean()],
                       width, label="|w_in|", color="steelblue")
        axes[0, 1].bar(x0 + width / 2,
                       [out_s[top_idx].mean(), out_s[rest_idx].mean()],
                       width, label="|w_out|", color="darkorange")
        axes[0, 1].set_xticks(x0); axes[0, 1].set_xticklabels(groups)
        axes[0, 1].set_title("|w_in| / |w_out| stratified by |w_fb|")
        axes[0, 1].legend(fontsize=8)

        # Bar chart of correlations (Pearson + Spearman)
        keys = list(corrs.keys())
        vals = [corrs[k] for k in keys]
        bar_colors = ["steelblue" if "in" in k else "darkorange" for k in keys]
        axes[0, 2].bar(range(len(keys)), vals, color=bar_colors)
        axes[0, 2].set_xticks(range(len(keys)))
        axes[0, 2].set_xticklabels(keys, rotation=25, ha="right", fontsize=8)
        axes[0, 2].axhline(0, color="gray", linewidth=0.5)
        axes[0, 2].set_ylim(-1, 1)
        axes[0, 2].set_title("|w_fb| vs |w_in|/|w_out| correlations")

        # Row 1: scatter plots with top-K highlighted
        axes[1, 0].scatter(fb_s, in_s, s=10, alpha=0.4, color="gray", label="all")
        axes[1, 0].scatter(fb_s[top_idx], in_s[top_idx], s=22, color="crimson",
                           label=f"top-{K} fb")
        axes[1, 0].set_xlabel("|w_fb|"); axes[1, 0].set_ylabel("|w_in|")
        axes[1, 0].set_title(
            f"|w_in| | ρ={corrs['spearman_fb_in']:+.2f}"); axes[1, 0].legend(fontsize=7)

        axes[1, 1].scatter(fb_s, out_s, s=10, alpha=0.4, color="gray")
        axes[1, 1].scatter(fb_s[top_idx], out_s[top_idx], s=22, color="crimson")
        axes[1, 1].set_xlabel("|w_fb|"); axes[1, 1].set_ylabel("|w_out|")
        axes[1, 1].set_title(
            f"|w_out| | ρ={corrs['spearman_fb_out']:+.2f}")

        # Quadrant heatmap
        axes[1, 2].imshow(quad_in, cmap="Reds", aspect="auto")
        axes[1, 2].set_xticks([0, 1]); axes[1, 2].set_xticklabels(["in: low", "in: high"])
        axes[1, 2].set_yticks([0, 1]); axes[1, 2].set_yticklabels(["fb: low", "fb: high"])
        for i in range(2):
            for j in range(2):
                axes[1, 2].text(j, i, f"{quad_in[i, j]}\n({quad_out[i, j]} out)",
                                ha="center", va="center", fontsize=9)
        axes[1, 2].set_title("median-split counts\n(in; out in parens)")

        plt.suptitle(f"{label}: do high-|w_fb| neurons also have high |w_in| or |w_out|?",
                     fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"fb_vs_in_out_{label}.png"),
                    dpi=200, bbox_inches="tight")
        plt.close()

        # ------------------------------------------------------------
        # (ii) Activity traces of top-K fb-heavy neurons on BOTH probes
        # ------------------------------------------------------------
        _, r_tri = forward_modified(m, x_tri, return_hidden=True)
        _, r_sq = forward_modified(m, x_sq, return_hidden=True)
        k_show = min(8, K)
        show_idx = top_idx[:k_show]

        fig, axes = plt.subplots(2, 2, figsize=(13, 6), sharex="col",
                                  gridspec_kw={"height_ratios": [1, 2]})
        # triangle column
        axes[0, 0].plot(x_tri.squeeze(-1), color="steelblue", label="input")
        axes[0, 0].plot(ref_tri.squeeze(-1), color="green", alpha=0.6, label="ref")
        axes[0, 0].axvspan(*win_tri["main"], color="red", alpha=0.05)
        axes[0, 0].axvspan(*win_tri["tail"], color="purple", alpha=0.05)
        axes[0, 0].set_title("triangle probe"); axes[0, 0].legend(fontsize=7)
        for idx in show_idx:
            axes[1, 0].plot(r_tri[:, idx], alpha=0.8,
                            label=f"n{idx} (|w_fb|={fb_s[idx]:.2f})")
        axes[1, 0].axvspan(*win_tri["main"], color="red", alpha=0.05)
        axes[1, 0].axvspan(*win_tri["tail"], color="purple", alpha=0.05)
        axes[1, 0].legend(fontsize=6, ncol=2); axes[1, 0].set_xlabel("t"); axes[1, 0].set_ylabel("r(t)")
        # square column
        axes[0, 1].plot(x_sq.squeeze(-1), color="steelblue", label="input")
        axes[0, 1].plot(ref_sq.squeeze(-1), color="green", alpha=0.6, label="ref")
        axes[0, 1].axvspan(*win_sq["main"], color="red", alpha=0.05)
        axes[0, 1].axvspan(*win_sq["post"], color="purple", alpha=0.05)
        axes[0, 1].set_title("square probe"); axes[0, 1].legend(fontsize=7)
        for idx in show_idx:
            axes[1, 1].plot(r_sq[:, idx], alpha=0.8)
        axes[1, 1].axvspan(*win_sq["main"], color="red", alpha=0.05)
        axes[1, 1].axvspan(*win_sq["post"], color="purple", alpha=0.05)
        axes[1, 1].set_xlabel("t")

        plt.suptitle(f"{label}: top-{k_show} |w_fb| neurons, triangle vs square",
                     fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"fb_heavy_traces_{label}.png"),
                    dpi=200, bbox_inches="tight")
        plt.close()

        # ------------------------------------------------------------
        # (iii) Peak-time analysis on the triangle probe
        # ------------------------------------------------------------
        # For each neuron (top-K + all), when does |r(t)| peak inside the
        # tail window? If the top-K tile the tail period, that's a temporal
        # basis / phase-code signature.
        t0, t1 = win_tri["tail"]
        peak_times_top = []
        for idx in top_idx:
            seg = np.abs(r_tri[t0:t1, idx])
            if seg.size:
                peak_times_top.append(t0 + int(np.argmax(seg)))
        peak_times_all = []
        for idx in range(n):
            seg = np.abs(r_tri[t0:t1, idx])
            if seg.size:
                peak_times_all.append(t0 + int(np.argmax(seg)))
        peak_rows = [
            {"model": label, "neuron": int(idx), "peak_t": int(pt),
             "|w_fb|": float(fb_s[idx]),
             "|w_in|": float(in_s[idx]),
             "|w_out|": float(out_s[idx]),
             "group": "top-K"}
            for idx, pt in zip(top_idx, peak_times_top)
        ]
        save_csv(peak_rows, os.path.join(out_dir, f"peak_times_{label}.csv"))

        # Raster: rows = top-K neurons sorted by peak time, color = r(t)
        sort_order = np.argsort(peak_times_top)
        sorted_idx = top_idx[sort_order]
        mat = np.stack([r_tri[:, idx] for idx in sorted_idx], axis=0)  # (K, T)
        # normalize each row for visibility
        mat_norm = mat / (np.max(np.abs(mat), axis=1, keepdims=True) + 1e-8)

        fig, axes = plt.subplots(2, 1, figsize=(8, 6),
                                  gridspec_kw={"height_ratios": [2, 1]}, sharex=True)
        im = axes[0].imshow(mat_norm, aspect="auto", cmap="RdBu_r",
                             vmin=-1, vmax=1,
                             extent=[0, mat.shape[1], K, 0])
        axes[0].axvspan(*win_tri["main"], color="red", alpha=0.1)
        axes[0].axvspan(*win_tri["tail"], color="purple", alpha=0.1)
        axes[0].set_ylabel("top-K neurons (sorted by tail peak time)")
        axes[0].set_title(f"{label}: top-{K} fb-heavy neurons, normalized r(t)")
        plt.colorbar(im, ax=axes[0], fraction=0.02)

        axes[1].hist(peak_times_top, bins=np.linspace(t0, t1, 20),
                      color="crimson", alpha=0.6, label="top-K fb")
        axes[1].hist(peak_times_all, bins=np.linspace(t0, t1, 20),
                      color="gray", alpha=0.4, label="all neurons")
        axes[1].axvspan(*win_tri["tail"], color="purple", alpha=0.05)
        axes[1].set_xlabel("time of tail-window peak")
        axes[1].set_ylabel("# neurons"); axes[1].legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"peak_time_raster_{label}.png"),
                    dpi=200, bbox_inches="tight")
        plt.close()

        # Aggregate per-neuron table
        table_rows = [{"neuron": i,
                       "w_fb": float(fb_s[i]),
                       "w_in": float(in_s[i]),
                       "w_out": float(out_s[i]),
                       "tail_peak_t": int(peak_times_all[i])}
                      for i in range(n)]
        save_csv(table_rows, os.path.join(out_dir, f"neuron_table_{label}.csv"))


# ============================================================
# SECTION B: ablations
# ============================================================
def section_B(models: Dict[str, LowRankRNN], save_dir: str) -> None:
    ensure_dir(save_dir)
    subset = {k: models[k] for k in ABLATION_MODELS if k in models}
    if not subset:
        print("[section_B] no ablation-target models present, skipping.")
        return
    B1_gain_sweep(subset, save_dir)
    B2_delay_sweep(subset, save_dir)
    B3_window_lesion(subset, save_dir)
    B4_fb_noise(subset, save_dir)
    B5_fb_freeze(subset, save_dir)
    B6_input_noise(subset, save_dir)
    B7_recurrent_noise(subset, save_dir)


def _plot_window_metric_curves(xvals, curves, xlabel, title, save_path,
                                ylabel="cosine"):
    """
    curves: {model_label: {window_name: [values by xval]}}
    Makes one subplot per window.
    """
    any_model = next(iter(curves))
    window_names = [w for w in curves[any_model].keys() if w != "all"]
    n_win = len(window_names)
    fig, axes = plt.subplots(1, n_win, figsize=(3.4 * n_win, 3.4), sharey=True)
    if n_win == 1:
        axes = [axes]
    for ax, w in zip(axes, window_names):
        for label in curves:
            ax.plot(xvals, curves[label][w], marker="o", label=label)
        ax.set_title(f"window: {w}"); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    plt.suptitle(title, fontsize=11)
    plt.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def B1_gain_sweep(models, save_dir):
    out_dir = os.path.join(save_dir, "B1_gain_sweep")
    ensure_dir(out_dir)
    gains = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    x_np, ref, windows = triangle_probe()

    curves = {lbl: {w: [] for w in windows} for lbl in models}
    rows = []
    for g in gains:
        for lbl, m in models.items():
            y, _ = forward_modified(m, x_np, fb_gain=g)
            wm = windowed_metrics(y, ref, windows)
            for w, d in wm.items():
                curves[lbl][w].append(d["cosine"])
            rows.append({"model": lbl, "gain": g,
                         **{f"cos_{w}": d["cosine"] for w, d in wm.items()}})
    save_csv(rows, os.path.join(out_dir, "metrics.csv"))
    _plot_window_metric_curves(gains, curves, xlabel="fb gain",
                                title="B1 — feedback gain sweep",
                                save_path=os.path.join(out_dir, "cosine_vs_gain.png"))

    # Example traces at three representative gains
    g_show = [0.0, 1.0, 2.0]
    fig, axes = plt.subplots(len(models), len(g_show),
                             figsize=(3.2 * len(g_show), 1.8 * len(models)),
                             sharex=True, sharey=True)
    if len(models) == 1:
        axes = axes[None, :]
    for r_i, (lbl, m) in enumerate(models.items()):
        for c_i, g in enumerate(g_show):
            y, _ = forward_modified(m, x_np, fb_gain=g)
            ax = axes[r_i, c_i]
            ax.plot(x_np.squeeze(-1), color="steelblue", alpha=0.4)
            ax.plot(ref.squeeze(-1), color="green", alpha=0.5)
            ax.plot(y.squeeze(-1), color="darkorange")
            if r_i == 0:
                ax.set_title(f"gain={g}", fontsize=9)
            if c_i == 0:
                ax.set_ylabel(lbl, fontsize=9)
    plt.suptitle("B1 — example traces")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "traces.png"), dpi=200, bbox_inches="tight")
    plt.close()


def B2_delay_sweep(models, save_dir):
    """
    Delay the feedback: at time t, feedback uses y(t-1-Δ) instead of y(t-1).
    Δ=0 is the original; Δ=1 is 'one-step late', etc.
    """
    out_dir = os.path.join(save_dir, "B2_delay_sweep")
    ensure_dir(out_dir)
    delays = [0, 1, 2, 5, 10, 20]
    x_np, ref, windows = triangle_probe()

    curves = {lbl: {w: [] for w in windows} for lbl in models}
    rows = []
    for d in delays:
        for lbl, m in models.items():
            y, _ = forward_modified(m, x_np, fb_delay=d)
            wm = windowed_metrics(y, ref, windows)
            for w, md in wm.items():
                curves[lbl][w].append(md["cosine"])
            rows.append({"model": lbl, "delay": d,
                         **{f"cos_{w}": md["cosine"] for w, md in wm.items()}})
    save_csv(rows, os.path.join(out_dir, "metrics.csv"))
    _plot_window_metric_curves(delays, curves, xlabel="fb delay (steps)",
                                title="B2 — feedback delay sweep",
                                save_path=os.path.join(out_dir, "cosine_vs_delay.png"))

    # Example traces at each delay (all delays, all models in one grid).
    fig, axes = plt.subplots(len(models), len(delays),
                             figsize=(2.6 * len(delays), 1.8 * len(models)),
                             sharex=True, sharey=True)
    if len(models) == 1:
        axes = axes[None, :]
    for r_i, (lbl, m) in enumerate(models.items()):
        for c_i, d in enumerate(delays):
            y, _ = forward_modified(m, x_np, fb_delay=d)
            ax = axes[r_i, c_i]
            ax.plot(x_np.squeeze(-1), color="steelblue", alpha=0.4)
            ax.plot(ref.squeeze(-1), color="green", alpha=0.5)
            ax.plot(y.squeeze(-1), color="darkorange")
            ax.axvspan(*windows["main"], color="red", alpha=0.05)
            ax.axvspan(*windows["tail"], color="purple", alpha=0.05)
            if r_i == 0:
                ax.set_title(f"Δ={d}", fontsize=9)
            if c_i == 0:
                ax.set_ylabel(lbl, fontsize=9)
    plt.suptitle("B2 — example traces across feedback delays")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "traces.png"), dpi=200, bbox_inches="tight")
    plt.close()


def B3_window_lesion(models, save_dir):
    """
    Zero out feedback during particular windows and see what degrades.
    Window definitions come from the probe (pre/main/tail).
    """
    out_dir = os.path.join(save_dir, "B3_window_lesion")
    ensure_dir(out_dir)
    x_np, ref, windows = triangle_probe()

    cases = ["none"] + [w for w in ["pre", "main", "tail"] if w in windows]
    rows = []
    fig, axes = plt.subplots(len(models), len(cases),
                             figsize=(3.0 * len(cases), 1.8 * len(models)),
                             sharex=True, sharey=True)
    if len(models) == 1:
        axes = axes[None, :]

    for r_i, (lbl, m) in enumerate(models.items()):
        for c_i, case in enumerate(cases):
            lesion = None if case == "none" else windows[case]
            y, _ = forward_modified(m, x_np, fb_lesion_window=lesion)
            wm = windowed_metrics(y, ref, windows)
            rows.append({"model": lbl, "lesion": case,
                         **{f"cos_{w}": d["cosine"] for w, d in wm.items()}})
            ax = axes[r_i, c_i]
            ax.plot(x_np.squeeze(-1), color="steelblue", alpha=0.4)
            ax.plot(ref.squeeze(-1), color="green", alpha=0.5)
            ax.plot(y.squeeze(-1), color="darkorange")
            if lesion is not None:
                ax.axvspan(*lesion, color="red", alpha=0.1)
            if r_i == 0:
                ax.set_title(f"lesion: {case}", fontsize=9)
            if c_i == 0:
                ax.set_ylabel(lbl, fontsize=9)
    plt.suptitle("B3 — time-windowed feedback lesion")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "traces.png"), dpi=200, bbox_inches="tight")
    plt.close()
    save_csv(rows, os.path.join(out_dir, "metrics.csv"))


def B4_fb_noise(models, save_dir):
    out_dir = os.path.join(save_dir, "B4_fb_noise")
    ensure_dir(out_dir)
    stds = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
    x_np, ref, windows = triangle_probe()
    n_repeats = 5

    curves = {lbl: {w: [] for w in windows} for lbl in models}
    curves_std = {lbl: {w: [] for w in windows} for lbl in models}
    rows = []
    for s in stds:
        for lbl, m in models.items():
            reps = {w: [] for w in windows}
            for _ in range(n_repeats):
                y, _ = forward_modified(m, x_np, fb_noise_std=s)
                wm = windowed_metrics(y, ref, windows)
                for w, d in wm.items():
                    reps[w].append(d["cosine"])
            for w in windows:
                curves[lbl][w].append(float(np.mean(reps[w])))
                curves_std[lbl][w].append(float(np.std(reps[w])))
            rows.append({"model": lbl, "fb_noise_std": s,
                         **{f"cos_{w}_mean": float(np.mean(reps[w])) for w in windows}})
    save_csv(rows, os.path.join(out_dir, "metrics.csv"))
    _plot_window_metric_curves(stds, curves, xlabel="fb noise std",
                                title="B4 — noise injected on feedback channel",
                                save_path=os.path.join(out_dir, "cosine_vs_fb_noise.png"))


def B5_fb_freeze(models, save_dir):
    """
    After the main triangle peak, clamp fb to a constant.
    If the tail emerges from the *live* feedback signal, this should kill it.
    """
    out_dir = os.path.join(save_dir, "B5_fb_freeze")
    ensure_dir(out_dir)
    x_np, ref, windows = triangle_probe()
    freeze_points = [windows["main"][0], (windows["main"][0] + windows["main"][1]) // 2,
                     windows["main"][1], windows["tail"][0] + 10]
    freeze_vals = [0.0, 0.5]
    rows = []
    fig, axes = plt.subplots(len(models), len(freeze_points) * len(freeze_vals),
                             figsize=(2.4 * len(freeze_points) * len(freeze_vals),
                                      1.8 * len(models)),
                             sharex=True, sharey=True)
    if len(models) == 1:
        axes = axes[None, :]

    col = 0
    col_titles = []
    for t_freeze in freeze_points:
        for v in freeze_vals:
            col_titles.append(f"t≥{t_freeze}, v={v}")
            for r_i, (lbl, m) in enumerate(models.items()):
                y, _ = forward_modified(m, x_np,
                                        fb_freeze_after=t_freeze,
                                        fb_freeze_value=v)
                wm = windowed_metrics(y, ref, windows)
                rows.append({"model": lbl, "freeze_after": t_freeze, "freeze_val": v,
                             **{f"cos_{w}": d["cosine"] for w, d in wm.items()}})
                ax = axes[r_i, col]
                ax.plot(x_np.squeeze(-1), color="steelblue", alpha=0.4)
                ax.plot(ref.squeeze(-1), color="green", alpha=0.5)
                ax.plot(y.squeeze(-1), color="darkorange")
                ax.axvline(t_freeze, linestyle=":", color="red", alpha=0.6)
                if r_i == 0:
                    ax.set_title(col_titles[-1], fontsize=7)
                if col == 0:
                    ax.set_ylabel(lbl, fontsize=9)
            col += 1
    plt.suptitle("B5 — feedback freeze")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "traces.png"), dpi=200, bbox_inches="tight")
    plt.close()
    save_csv(rows, os.path.join(out_dir, "metrics.csv"))


def B6_input_noise(models, save_dir):
    """Does feedback buy robustness to *input* noise?"""
    out_dir = os.path.join(save_dir, "B6_input_noise")
    ensure_dir(out_dir)
    stds = [0.0, 0.01, 0.025, 0.05, 0.1, 0.2]
    x_np, ref, windows = triangle_probe()
    n_repeats = 5

    curves = {lbl: {w: [] for w in windows} for lbl in models}
    rows = []
    for s in stds:
        for lbl, m in models.items():
            reps = {w: [] for w in windows}
            for _ in range(n_repeats):
                y, _ = forward_modified(m, x_np, input_noise_std=s)
                wm = windowed_metrics(y, ref, windows)
                for w, d in wm.items():
                    reps[w].append(d["cosine"])
            for w in windows:
                curves[lbl][w].append(float(np.mean(reps[w])))
            rows.append({"model": lbl, "input_noise_std": s,
                         **{f"cos_{w}_mean": float(np.mean(reps[w])) for w in windows}})
    save_csv(rows, os.path.join(out_dir, "metrics.csv"))
    _plot_window_metric_curves(stds, curves, xlabel="input noise std",
                                title="B6 — input-noise robustness (FB vs noFB)",
                                save_path=os.path.join(out_dir, "cosine_vs_input_noise.png"))


def B7_recurrent_noise(models, save_dir):
    """Does feedback buy robustness to intrinsic recurrent noise?"""
    out_dir = os.path.join(save_dir, "B7_recurrent_noise")
    ensure_dir(out_dir)
    stds = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
    x_np, ref, windows = triangle_probe()
    n_repeats = 5

    curves = {lbl: {w: [] for w in windows} for lbl in models}
    rows = []
    for s in stds:
        for lbl, m in models.items():
            reps = {w: [] for w in windows}
            for _ in range(n_repeats):
                y, _ = forward_modified(m, x_np, rec_noise_std=s)
                wm = windowed_metrics(y, ref, windows)
                for w, d in wm.items():
                    reps[w].append(d["cosine"])
            for w in windows:
                curves[lbl][w].append(float(np.mean(reps[w])))
            rows.append({"model": lbl, "rec_noise_std": s,
                         **{f"cos_{w}_mean": float(np.mean(reps[w])) for w in windows}})
    save_csv(rows, os.path.join(out_dir, "metrics.csv"))
    _plot_window_metric_curves(stds, curves, xlabel="recurrent noise std",
                                title="B7 — recurrent-noise robustness (FB vs noFB)",
                                save_path=os.path.join(out_dir, "cosine_vs_rec_noise.png"))


# ============================================================
# SECTION C: dynamical-systems analysis
# ============================================================
def section_C(models: Dict[str, LowRankRNN], save_dir: str) -> None:
    ensure_dir(save_dir)
    C2_jacobian_eigenvalues(models, save_dir)
    C3_effective_dimensionality(models, save_dir)
    C4_future_decoder(models, save_dir)


@torch.no_grad()
def _run_and_collect(model: LowRankRNN, x_np: np.ndarray):
    """Run a model on an input and return numpy arrays for h, r, output."""
    x = torch.from_numpy(x_np).float().to(device).unsqueeze(0)
    y, h, r, rec, inp, fb = model(x, return_dynamics=True)
    return (y.squeeze(0).cpu().numpy(),
            h.squeeze(0).cpu().numpy(),
            r.squeeze(0).cpu().numpy())


def C2_jacobian_eigenvalues(models, save_dir):
    """
    Linearize the update around each visited state and plot the
    nontrivial (rank-r) eigenvalues in the complex plane, colored by time.
    Look for complex-conjugate pairs with |λ| ≈ 1 during the tail window —
    that's an oscillator.
    """
    out_dir = os.path.join(save_dir, "C2_jacobian")
    ensure_dir(out_dir)
    x_np, ref, windows = triangle_probe()

    for label, m in models.items():
        y_np, h_np, r_np = _run_and_collect(m, x_np)
        Mm = m.M.detach().cpu().numpy()
        Nn = m.N.detach().cpu().numpy()
        alpha = m.alpha
        T, H = r_np.shape

        # Nonzero eigenvalues of (M N^T / H) diag(1 - tanh^2(h)) are those
        # of (N^T / H) diag(1-tanh^2) M, which is rank-r × rank-r. Adding
        # the (1-alpha)I contribution shifts the full-matrix spectrum, but
        # the "dynamical" eigenvalues of interest are
        #   mu = (1 - alpha) + alpha * lambda_small
        # where lambda_small come from the r×r reduced matrix.
        eigs_over_time = np.zeros((T, m.rank), dtype=complex)
        for t in range(T):
            D = 1.0 - r_np[t] ** 2             # (H,)
            # reduced r×r matrix
            B = (Nn.T * D) @ Mm / m.n_hidden   # (rank, rank)
            lam = np.linalg.eigvals(B)
            # full-system eigenvalues from the low-rank part
            mu = (1 - alpha) + alpha * lam
            eigs_over_time[t] = mu

        # plot in complex plane, color by time window
        fig, ax = plt.subplots(figsize=(5.5, 5))
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), "k--", linewidth=0.6, alpha=0.5)
        ax.axhline(0, color="gray", linewidth=0.5); ax.axvline(0, color="gray", linewidth=0.5)

        window_colors = {"pre": "gray", "main": "crimson", "tail": "purple"}
        for wname in ["pre", "main", "tail"]:
            t0, t1 = windows[wname]
            sub = eigs_over_time[t0:t1].reshape(-1)
            ax.scatter(sub.real, sub.imag, s=14, alpha=0.5,
                       color=window_colors[wname], label=wname)
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_aspect("equal")
        ax.set_title(f"{label}: low-rank Jacobian eigenvalues over time")
        ax.set_xlabel("Re(μ)"); ax.set_ylabel("Im(μ)")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"eigs_{label}.png"), dpi=200, bbox_inches="tight")
        plt.close()

        # |μ| and phase over time
        fig, axes = plt.subplots(2, 1, figsize=(8, 4.5), sharex=True)
        for k in range(m.rank):
            axes[0].plot(np.abs(eigs_over_time[:, k]), label=f"|μ_{k}|")
            axes[1].plot(np.imag(eigs_over_time[:, k]), label=f"Im(μ_{k})")
        for ax in axes:
            ax.axvspan(*windows["main"], color="red", alpha=0.05)
            ax.axvspan(*windows["tail"], color="purple", alpha=0.05)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("|μ|"); axes[1].set_ylabel("Im(μ)")
        axes[1].set_xlabel("t")
        axes[0].set_title(f"{label}: eigenvalue magnitudes / imaginary parts")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"eigs_vs_time_{label}.png"),
                    dpi=200, bbox_inches="tight")
        plt.close()


def C3_effective_dimensionality(models, save_dir):
    """
    Participation ratio of the activity covariance per time window.
    PR = (sum λ)^2 / sum λ^2 = trace(C)^2 / trace(C @ C).
    """
    out_dir = os.path.join(save_dir, "C3_effective_dim")
    ensure_dir(out_dir)

    # Collect activity from a small battery of probes.
    probes = [
        ("triangle_big", *triangle_probe(tri_amp=1.3, tri_center=170)),
        ("triangle_mid", *triangle_probe(tri_amp=1.0, tri_center=170)),
        ("seq_sq_tri", *(lambda p: (p[0], p[1], None))(
            make_probe_sequential(T=CFG.probe_T, sq_center=50, tri_center=170))),
    ]
    # For the seq probe we don't have auto windows, skip it from the
    # window-split metric and only compute the "all" PR.
    rows = []
    for label, m in models.items():
        for pname, x_np, ref, windows_or_none in probes:
            _, _, r_np = _run_and_collect(m, x_np)
            if windows_or_none is None:
                winset = {"all": (0, r_np.shape[0])}
            else:
                winset = windows_or_none
            for wname, (t0, t1) in winset.items():
                sub = r_np[t0:t1]
                if sub.shape[0] < 3:
                    continue
                sub = sub - sub.mean(axis=0, keepdims=True)
                C = (sub.T @ sub) / max(1, sub.shape[0] - 1)
                num = np.trace(C) ** 2
                den = np.trace(C @ C) + 1e-12
                pr = float(num / den)
                rows.append({"model": label, "probe": pname, "window": wname,
                             "participation_ratio": pr})

    save_csv(rows, os.path.join(out_dir, "pr.csv"))

    # Summary plot: PR by (model, window) averaged over probes
    model_labels = list(models.keys())
    window_names = ["pre", "main", "tail"]
    mean_table = np.zeros((len(model_labels), len(window_names)))
    for i, lbl in enumerate(model_labels):
        for j, w in enumerate(window_names):
            vals = [r["participation_ratio"] for r in rows
                    if r["model"] == lbl and r["window"] == w]
            mean_table[i, j] = float(np.mean(vals)) if vals else np.nan

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(model_labels))
    width = 0.25
    for j, w in enumerate(window_names):
        ax.bar(x + (j - 1) * width, mean_table[:, j], width, label=w)
    ax.set_xticks(x); ax.set_xticklabels(model_labels, rotation=20, ha="right")
    ax.set_ylabel("Participation ratio (activity)")
    ax.set_title("C3 — effective dimensionality per window")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pr_by_window.png"), dpi=200, bbox_inches="tight")
    plt.close()


def C4_future_decoder(models, save_dir):
    """
    Train a ridge regressor: r(t) → y(t+k). If feedback carries 'intent',
    FB should be decodable at larger k than noFB.
    """
    out_dir = os.path.join(save_dir, "C4_future_decoder")
    ensure_dir(out_dir)

    # Build a training + testing dataset of (r_t, y_{t+k}) pairs across many probes.
    tri_centers_train = [120, 140, 160, 180, 200]
    tri_centers_test = [130, 170, 210]
    tri_amps = [0.9, 1.1, 1.3]
    ks = [1, 5, 10, 20, 30, 50]

    def collect(model, centers):
        Rs, Ys = [], []
        for c in centers:
            for a in tri_amps:
                x_np, ref, _ = triangle_probe(tri_amp=a, tri_center=c)
                y_np, _, r_np = _run_and_collect(model, x_np)
                Rs.append(r_np)           # (T, H)
                Ys.append(y_np.squeeze(-1))  # (T,)
        return Rs, Ys

    rows = []
    r2_table = {lbl: [] for lbl in models}
    for lbl, m in models.items():
        R_tr_list, Y_tr_list = collect(m, tri_centers_train)
        R_te_list, Y_te_list = collect(m, tri_centers_test)
        for k in ks:
            Xtr, Ttr = [], []
            for R, Y in zip(R_tr_list, Y_tr_list):
                if R.shape[0] <= k:
                    continue
                Xtr.append(R[:-k])
                Ttr.append(Y[k:])
            Xte, Tte = [], []
            for R, Y in zip(R_te_list, Y_te_list):
                if R.shape[0] <= k:
                    continue
                Xte.append(R[:-k])
                Tte.append(Y[k:])
            Xtr = np.concatenate(Xtr, 0); Ttr = np.concatenate(Ttr, 0)
            Xte = np.concatenate(Xte, 0); Tte = np.concatenate(Tte, 0)

            reg = Ridge(alpha=1.0).fit(Xtr, Ttr)
            pred = reg.predict(Xte)
            ss_res = float(((pred - Tte) ** 2).sum())
            ss_tot = float(((Tte - Tte.mean()) ** 2).sum()) + 1e-12
            r2 = 1.0 - ss_res / ss_tot
            r2_table[lbl].append(r2)
            rows.append({"model": lbl, "k": k, "test_r2": r2})

    save_csv(rows, os.path.join(out_dir, "r2.csv"))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for lbl in r2_table:
        ax.plot(ks, r2_table[lbl], marker="o", label=lbl)
    ax.set_xlabel("forecast horizon k (steps)")
    ax.set_ylabel("test R² (r_t → y_{t+k})")
    ax.set_title("C4 — linear decodability of future output from hidden state")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "r2_vs_horizon.png"), dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================
# Entry point
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", default=CFG.base_dir,
                   help="directory containing with_feedback/ and no_feedback/ subdirs")
    p.add_argument("--sections", default="ABC",
                   help="which sections to run, e.g. 'A', 'AB', 'ABC'")
    args = p.parse_args()

    set_seed(ANALYSIS_SEED)
    out_root = os.path.join(args.base_dir, "analysis")
    ensure_dir(out_root)

    print(f"Loading checkpoints from {args.base_dir} ...")
    models = load_models(args.base_dir)
    print(f"Loaded models: {list(models.keys())}")

    if "A" in args.sections:
        print("\n=== Section A: weight-level ===")
        section_A(models, os.path.join(out_root, "A"))
    if "B" in args.sections:
        print("\n=== Section B: ablations ===")
        section_B(models, os.path.join(out_root, "B"))
    if "C" in args.sections:
        print("\n=== Section C: dynamics ===")
        section_C(models, os.path.join(out_root, "C"))

    print(f"\nAll analysis outputs under {out_root}")


if __name__ == "__main__":
    main()
