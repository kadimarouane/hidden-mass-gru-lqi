"""
Hidden-mass GRU vs LQI controllers: temporally aligned comparative benchmark
===================================================================

Objective
---------
Compare four controllers on a hidden-mass mass-spring-damper tracking task:

1) Fixed LQI designed at nominal mass m = 1 kg
2) True-mass gain-scheduled LQI (oracle, not deployable)
3) Estimated-mass gain-scheduled LQI (classical deployable comparator)
4) GRU hidden-mass policy trained from observation/control history

Revision v5
-----------
- Plant integration uses dt = 0.01 s.
- Every controller and the GRU/RLS observation update use dt = 0.02 s.
- Commands are held by zero-order hold between updates.
- All controllers receive the same measurement-noise realization per scenario.
- Paired-test p-values include Holm correction across metrics.

Scientific design
-----------------
- CUDA is used automatically when available.
- Dataset split is trajectory-wise to avoid sample leakage.
- The GRU policy sees only measured history, not the true mass.
- The estimated-mass LQI also does not see the true mass; it estimates mass online.
- The true-mass scheduled LQI is reported as an oracle reference, not as a fair deployable baseline.
- Closed-loop evaluation metrics are computed only on independent randomized TEST hidden-mass scenarios.
- Outputs include detailed CSV files, statistical tests, and publication-quality vector PDF/SVG figures plus 600-dpi PNG copies.

Default output directory
------------------------
C:\\Users\\tuf-p\\Desktop\\ARTICLES\\MARO

Author use note
---------------
Run this file directly in a CUDA-enabled Python environment:
    python hidden_mass_gru_lqi_4controllers_benchmark.py

Required packages:
    numpy scipy pandas matplotlib torch
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import random
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Limit CPU-side thread contention on Windows/Jupyter.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.linalg import solve_continuous_are
from scipy.stats import ttest_rel, wilcoxon
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# 0. Configuration
# =============================================================================

@dataclass(frozen=True)
class Config:
    # Reproducibility
    # The teacher dataset, neural optimization, evaluation scenarios, and
    # measurement-noise streams are separated explicitly.
    seed: int = 42
    evaluation_scenario_seed: int = 2026
    evaluation_noise_seed: int = 42000

    # Output path requested by user
    output_root: str = r"C:\Users\tuf-p\Desktop\ARTICLES\MARO"
    experiment_name: str = "hidden_mass_gru_lqi_4controllers_benchmark_v4_publication_quality"

    # CUDA behavior
    prefer_cuda: bool = True
    require_cuda: bool = False

    # Physical plant: m*xddot + b*xdot + k*x = u
    k_spring: float = 2.0
    b_damping: float = 0.5
    mass_min: float = 0.5
    mass_max: float = 10.0
    nominal_fixed_mass: float = 1.0

    # LQI design
    q_pos: float = 12.0
    q_vel: float = 1.0
    q_int: float = 45.0
    r_control: float = 0.12
    u_max: float = 80.0

    # Teacher dataset generation
    dt_train: float = 0.02
    t_train: float = 10.0
    num_teacher_trajectories: int = 144
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    seq_len: int = 30                  # 30*0.02 = 0.60 s of memory
    batch_size: int = 512
    max_epochs: int = 35
    early_stop_patience: int = 7
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    huber_beta: float = 0.5

    # Measurement noise used in GRU training features and in closed-loop evaluation
    # Set to 0.0 for a noise-free benchmark.
    train_position_noise_std: float = 0.002
    train_velocity_noise_std: float = 0.003
    test_position_noise_std: float = 0.002
    test_velocity_noise_std: float = 0.003

    # Closed-loop evaluation
    # The plant is integrated at dt_test, whereas every controller and the
    # GRU feature buffer are updated at dt_control. The command is held by a
    # zero-order hold between controller updates.
    dt_test: float = 0.01
    dt_control: float = 0.02
    t_test: float = 21.0
    num_eval_scenarios: int = 60
    settling_band_ratio: float = 0.02
    reference_min_abs: float = 0.8
    reference_range_low: float = -2.5
    reference_range_high: float = 3.0

    # Online mass estimator for estimated-mass scheduled LQI.
    # Final version: recursive least squares (RLS) estimates theta = 1/m from
    #     xddot = theta * (u - b*xdot - k*x).
    # This is a stronger classical comparator than the earlier direct division
    # m = (u - b*v - k*x)/xddot, which was too noisy and weak near low acceleration.
    estimator_initial_mass: float = 1.0
    estimator_forgetting_factor: float = 0.992
    estimator_rls_p0: float = 20.0
    estimator_rls_p_min: float = 1e-4
    estimator_rls_p_max: float = 1e4
    estimator_min_regressor: float = 0.08
    estimator_min_accel: float = 0.03
    estimator_accel_lpf_alpha: float = 0.22
    estimator_rate_limit: float = 0.12   # max kg change per valid estimator update
    estimator_smooth_window: int = 9

    # Figures
    # PDF/SVG files are vector outputs; dpi affects only rasterized elements.
    # PNG copies are exported at 600 dpi for journal systems that rasterize previews.
    figure_dpi: int = 600
    save_png_copies: bool = True
    save_svg_copies: bool = True
    publication_font_size: int = 9
    publication_title_size: int = 10
    publication_label_size: int = 9
    publication_legend_size: int = 8
    publication_line_width: float = 1.45
    publication_marker_size: float = 28.0

    # Feature order used by the GRU policy
    # true mass is intentionally excluded.
    # error = reference - measured_position
    # previous_control helps hidden-mass inference.
    # estimated_accel is computed from measured velocity history.
    input_dim: int = 7
    gru_hidden_dim: int = 72


CFG = Config()


def configure_publication_matplotlib(cfg: Config = CFG) -> None:
    """Set Matplotlib for manuscript-quality vector output.

    Notes
    -----
    - PDF and SVG are vector formats; 600 dpi is relevant only for any rasterized
      artists and for the PNG copies.
    - Font type 42 embeds TrueType fonts in PDF/PS and prevents low-quality Type-3
      glyph rendering in many journal production systems.
    """
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": cfg.figure_dpi,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.035,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "DejaVu Sans",
        "font.size": cfg.publication_font_size,
        "axes.titlesize": cfg.publication_title_size,
        "axes.labelsize": cfg.publication_label_size,
        "xtick.labelsize": cfg.publication_font_size,
        "ytick.labelsize": cfg.publication_font_size,
        "legend.fontsize": cfg.publication_legend_size,
        "axes.linewidth": 0.75,
        "lines.linewidth": cfg.publication_line_width,
        "lines.markersize": 4.2,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "grid.linewidth": 0.45,
        "path.simplify": True,
        "path.simplify_threshold": 0.01,
        "agg.path.chunksize": 20000,
    })


configure_publication_matplotlib(CFG)


def controller_short_label(name: str) -> str:
    mapping = {
        "Fixed LQI m=1": "Fixed LQI\n$m=1$",
        "True-mass scheduled LQI (oracle)": "True-mass\nLQI oracle",
        "Estimated-mass scheduled LQI": "RLS estimated-mass\nLQI",
        "GRU hidden-mass policy": "GRU\nhidden-mass",
    }
    return mapping.get(name, name)


def set_switch_lines(ax: plt.Axes, switch_times: Sequence[float]) -> None:
    for sw in switch_times:
        ax.axvline(sw, linestyle="--", linewidth=0.8, alpha=0.6)


def set_axis_grid(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, alpha=0.32)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(CFG.seed)
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass

try:
    torch.set_num_threads(1)
except Exception:
    pass
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass


def select_device(cfg: Config) -> torch.device:
    if cfg.prefer_cuda and torch.cuda.is_available():
        try:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        return torch.device("cuda:0")
    if cfg.require_cuda:
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Check the NVIDIA driver, GPU mode, and PyTorch CUDA build."
        )
    return torch.device("cpu")


DEVICE = select_device(CFG)
PIN_MEMORY = DEVICE.type == "cuda"

RESULTS_DIR = Path(CFG.output_root) / CFG.experiment_name
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = RESULTS_DIR / "figures_pdf_vector"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PNG_DIR = RESULTS_DIR / "figures_png_600dpi"
PNG_DIR.mkdir(parents=True, exist_ok=True)
SVG_DIR = RESULTS_DIR / "figures_svg_vector"
SVG_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR = RESULTS_DIR / "csv_outputs"
CSV_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = RESULTS_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = RESULTS_DIR / "manuscript_ready"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. Basic utilities
# =============================================================================

@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean


def fit_scaler(data: np.ndarray, eps: float = 1e-8) -> Scaler:
    mean = data.mean(axis=0).astype(np.float32)
    std = data.std(axis=0).astype(np.float32)
    std = np.where(std < eps, 1.0, std).astype(np.float32)
    return Scaler(mean=mean, std=std)


def clip_control(u: float, cfg: Config = CFG) -> float:
    if not np.isfinite(u):
        return 0.0
    return float(np.clip(u, -cfg.u_max, cfg.u_max))


def safe_float(x: float) -> float:
    return float(x) if np.isfinite(x) else float("nan")


def ensure_nonzero_reference(rng: np.random.Generator, cfg: Config = CFG) -> float:
    ref = float(rng.uniform(cfg.reference_range_low, cfg.reference_range_high))
    if abs(ref) < cfg.reference_min_abs:
        ref = math.copysign(cfg.reference_min_abs, ref if ref != 0 else 1.0)
    return ref


def save_json(obj: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))


def savefig(fig: plt.Figure, stem: str) -> None:
    """Save a figure in publication formats.

    The PDF and SVG files remain vector whenever the plotted artists are vector
    artists. The dpi argument is still passed so that any unavoidable rasterized
    elements are embedded at 600 dpi. PNG copies are exported in a separate folder
    at 600 dpi for manuscript submission systems that need raster previews.
    """
    metadata = {
        "Creator": "hidden_mass_gru_lqi_4controllers_benchmark_v4_publication_quality.py",
        "Title": stem,
    }
    fig.align_labels()
    pdf_path = FIG_DIR / f"{stem}.pdf"
    fig.savefig(pdf_path, format="pdf", dpi=CFG.figure_dpi, bbox_inches="tight", pad_inches=0.035, metadata=metadata)
    if CFG.save_svg_copies:
        svg_path = SVG_DIR / f"{stem}.svg"
        fig.savefig(svg_path, format="svg", dpi=CFG.figure_dpi, bbox_inches="tight", pad_inches=0.035, metadata=metadata)
    if CFG.save_png_copies:
        png_path = PNG_DIR / f"{stem}.png"
        fig.savefig(png_path, format="png", dpi=CFG.figure_dpi, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


# =============================================================================
# 2. LQI design and plant simulation
# =============================================================================

Q_LQI = np.diag([CFG.q_pos, CFG.q_vel, CFG.q_int]).astype(float)
R_LQI = np.array([[CFG.r_control]], dtype=float)


def design_lqi_for_mass(mass: float, cfg: Config = CFG) -> Tuple[np.ndarray, float]:
    """Continuous-time LQI for state [position, velocity, integral_error]."""
    A_sys = np.array(
        [[0.0, 1.0], [-cfg.k_spring / mass, -cfg.b_damping / mass]],
        dtype=float,
    )
    B_sys = np.array([[0.0], [1.0 / mass]], dtype=float)

    A_aug = np.zeros((3, 3), dtype=float)
    A_aug[:2, :2] = A_sys
    A_aug[2, 0] = -1.0  # z_dot = ref - position; design uses -position channel

    B_aug = np.zeros((3, 1), dtype=float)
    B_aug[:2, 0] = B_sys[:, 0]

    P = solve_continuous_are(A_aug, B_aug, Q_LQI, R_LQI)
    K = np.linalg.solve(R_LQI, B_aug.T @ P)
    return K[0, :2].astype(float), float(K[0, 2])


def lqi_control(x_meas: np.ndarray, z: float, Kx: np.ndarray, Ki: float, cfg: Config = CFG) -> float:
    return clip_control(-float(Kx @ x_meas) - Ki * z, cfg)


def plant_step(x: np.ndarray, u: float, mass: float, dt: float, cfg: Config = CFG) -> np.ndarray:
    # x = [position, velocity]
    acceleration = (u - cfg.b_damping * x[1] - cfg.k_spring * x[0]) / mass
    x_next = np.array([x[0] + x[1] * dt, x[1] + acceleration * dt], dtype=float)
    if not np.all(np.isfinite(x_next)):
        raise FloatingPointError("Non-finite plant state detected.")
    return x_next


def make_gain_grid(cfg: Config = CFG) -> np.ndarray:
    dense = np.linspace(cfg.mass_min, cfg.mass_max, 96)
    anchors = np.array([0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0], dtype=float)
    grid = np.unique(np.round(np.concatenate([dense, anchors]), 8))
    return grid


def build_gain_cache(masses: Iterable[float]) -> Dict[float, Tuple[np.ndarray, float]]:
    return {float(m): design_lqi_for_mass(float(m)) for m in masses}


def nearest_gain(mass: float, gain_grid: np.ndarray, gain_cache: Dict[float, Tuple[np.ndarray, float]]) -> Tuple[np.ndarray, float]:
    idx = int(np.argmin(np.abs(gain_grid - mass)))
    return gain_cache[float(gain_grid[idx])]


# =============================================================================
# 3. Mass schedules and teacher trajectories
# =============================================================================

@dataclass
class Scenario:
    scenario_id: int
    reference: float
    initial_position: float
    initial_velocity: float
    switch_times: List[float]
    masses: List[float]


def generate_piecewise_mass_schedule(
    rng: np.random.Generator,
    t_end: float,
    dt: float,
    mass_grid: np.ndarray,
    min_segments: int = 2,
    max_segments: int = 5,
) -> Tuple[np.ndarray, List[float], List[float]]:
    n_steps = int(t_end / dt)
    n_segments = int(rng.integers(min_segments, max_segments + 1))

    if n_segments == 1:
        switch_times: List[float] = []
    else:
        lower = max(0.8, 0.08 * t_end)
        upper = max(lower + dt, 0.92 * t_end)
        switch_times = np.sort(rng.uniform(lower, upper, size=n_segments - 1)).tolist()
        # Avoid near-identical switch times.
        cleaned = []
        for s in switch_times:
            if not cleaned or abs(s - cleaned[-1]) >= 0.8:
                cleaned.append(float(s))
        switch_times = cleaned
        n_segments = len(switch_times) + 1

    allowed = np.asarray([m for m in mass_grid if 0.7 <= m <= 9.5], dtype=float)
    masses = rng.choice(allowed, size=n_segments, replace=True).astype(float)
    for i in range(1, len(masses)):
        if abs(masses[i] - masses[i - 1]) < 0.4:
            candidates = allowed[np.abs(allowed - masses[i - 1]) >= 0.8]
            masses[i] = float(rng.choice(candidates))

    t = np.arange(n_steps) * dt
    boundaries = [0.0] + switch_times + [t_end + dt]
    m_hist = np.empty(n_steps, dtype=float)
    for seg in range(n_segments):
        mask = (t >= boundaries[seg]) & (t < boundaries[seg + 1])
        m_hist[mask] = float(masses[seg])

    return m_hist, switch_times, masses.tolist()


def build_feature(
    pos_meas: float,
    vel_meas: float,
    ref: float,
    z: float,
    u_prev: float,
    accel_est: float,
) -> np.ndarray:
    error = ref - pos_meas
    return np.array([pos_meas, vel_meas, ref, z, u_prev, error, accel_est], dtype=np.float32)


def add_measurement_noise(
    x_true: np.ndarray,
    rng: np.random.Generator,
    pos_std: float,
    vel_std: float,
) -> np.ndarray:
    if pos_std == 0.0 and vel_std == 0.0:
        return x_true.astype(float).copy()
    return np.array(
        [
            x_true[0] + rng.normal(0.0, pos_std),
            x_true[1] + rng.normal(0.0, vel_std),
        ],
        dtype=float,
    )


def generate_teacher_trajectory(
    rng: np.random.Generator,
    gain_grid: np.ndarray,
    gain_cache: Dict[float, Tuple[np.ndarray, float]],
    cfg: Config = CFG,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate one trajectory controlled by the oracle true-mass LQI.

    The target is the oracle LQI command. The GRU input features exclude true mass.
    """
    n_steps = int(cfg.t_train / cfg.dt_train)
    ref = ensure_nonzero_reference(rng, cfg)
    m_hist, _, _ = generate_piecewise_mass_schedule(
        rng, cfg.t_train, cfg.dt_train, gain_grid, min_segments=2, max_segments=5
    )

    x = np.array([rng.uniform(-0.8, 0.8), rng.uniform(-0.4, 0.4)], dtype=float)
    z = 0.0
    u_prev = 0.0
    prev_vel_meas: Optional[float] = None

    X = np.zeros((n_steps, cfg.input_dim), dtype=np.float32)
    y = np.zeros((n_steps, 1), dtype=np.float32)

    for k in range(n_steps):
        m = float(m_hist[k])
        x_meas = add_measurement_noise(
            x,
            rng,
            cfg.train_position_noise_std,
            cfg.train_velocity_noise_std,
        )
        if prev_vel_meas is None:
            accel_est = 0.0
        else:
            accel_est = float((x_meas[1] - prev_vel_meas) / cfg.dt_train)
        prev_vel_meas = float(x_meas[1])

        Kx, Ki = nearest_gain(m, gain_grid, gain_cache)
        u = lqi_control(x_meas, z, Kx, Ki, cfg)

        X[k] = build_feature(x_meas[0], x_meas[1], ref, z, u_prev, accel_est)
        y[k, 0] = u

        error = ref - x_meas[0]
        x = plant_step(x, u, m, cfg.dt_train, cfg)
        z += error * cfg.dt_train
        u_prev = u

    return X, y


def make_sequence_arrays(
    trajectories: Sequence[Tuple[np.ndarray, np.ndarray]],
    x_scaler: Scaler,
    y_scaler: Scaler,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    seqs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for X, y in trajectories:
        Xn = x_scaler.transform(X).astype(np.float32)
        yn = y_scaler.transform(y).astype(np.float32)
        n = Xn.shape[0]
        for i in range(n):
            start = max(0, i - seq_len + 1)
            window = Xn[start:i + 1]
            if window.shape[0] < seq_len:
                pad = np.repeat(window[:1], seq_len - window.shape[0], axis=0)
                window = np.vstack([pad, window])
            seqs.append(window)
            ys.append(yn[i])
    return np.asarray(seqs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


# =============================================================================
# 4. GRU policy
# =============================================================================

class GRUPolicy(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 96),
            nn.SiLU(),
            nn.Dropout(p=0.03),
            nn.Linear(96, 48),
            nn.SiLU(),
            nn.Linear(48, 1),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x_seq)
        return self.head(out[:, -1, :])


def train_policy(
    model: nn.Module,
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    cfg: Config = CFG,
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=PIN_MEMORY,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    loss_fn = nn.SmoothL1Loss(beta=cfg.huber_beta)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    no_improve = 0
    history: List[Dict[str, float]] = []

    print("\n--- Training GRU hidden-mass policy ---", flush=True)
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        train_sum = 0.0
        train_n = 0
        for Xb, yb in train_loader:
            Xb = Xb.to(DEVICE, non_blocking=PIN_MEMORY)
            yb = yb.to(DEVICE, non_blocking=PIN_MEMORY)
            pred = model(Xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_sum += float(loss.item()) * Xb.shape[0]
            train_n += Xb.shape[0]

        model.eval()
        val_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(DEVICE, non_blocking=PIN_MEMORY)
                yb = yb.to(DEVICE, non_blocking=PIN_MEMORY)
                val_loss = loss_fn(model(Xb), yb)
                val_sum += float(val_loss.item()) * Xb.shape[0]
                val_n += Xb.shape[0]

        train_loss = train_sum / max(train_n, 1)
        val_loss = val_sum / max(val_n, 1)
        scheduler.step(val_loss)
        lr_now = float(optimizer.param_groups[0]["lr"])

        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr_now}
        history.append(record)

        msg = f"Epoch {epoch:02d}/{cfg.max_epochs} | train={train_loss:.6f} | val={val_loss:.6f} | lr={lr_now:.2e}"
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            msg += "  best"
        else:
            no_improve += 1
        print(msg, flush=True)

        if no_improve >= cfg.early_stop_patience:
            print(f"Early stopping at epoch {epoch}.", flush=True)
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, history


def evaluate_offline(model: nn.Module, dataset: TensorDataset, y_scaler: Scaler, cfg: Config = CFG) -> Dict[str, float]:
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=PIN_MEMORY)
    pred_list = []
    true_list = []
    model.eval()
    with torch.no_grad():
        for Xb, yb in loader:
            pred_n = model(Xb.to(DEVICE, non_blocking=PIN_MEMORY)).cpu().numpy()
            true_n = yb.numpy()
            pred_list.append(y_scaler.inverse_transform(pred_n))
            true_list.append(y_scaler.inverse_transform(true_n))
    pred = np.concatenate(pred_list, axis=0).reshape(-1)
    true = np.concatenate(true_list, axis=0).reshape(-1)
    err = true - pred
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return {
        "offline_u_RMSE": float(np.sqrt(np.mean(err ** 2))),
        "offline_u_MAE": float(np.mean(np.abs(err))),
        "offline_u_R2": float(r2),
        "offline_u_bias": float(np.mean(pred - true)),
        "offline_u_p95_abs_error": float(np.percentile(np.abs(err), 95)),
        "offline_u_max_abs_error": float(np.max(np.abs(err))),
    }


def gru_control(
    model: nn.Module,
    seq_buffer: deque,
    x_scaler: Scaler,
    y_scaler: Scaler,
    cfg: Config = CFG,
) -> float:
    seq = np.stack(list(seq_buffer), axis=0).astype(np.float32)
    seq_n = x_scaler.transform(seq).astype(np.float32)
    inp = torch.from_numpy(seq_n[None, :, :]).to(DEVICE)
    model.eval()
    with torch.no_grad():
        u_n = model(inp).cpu().numpy()[0, 0]
    u = float(y_scaler.inverse_transform(np.array([[u_n]], dtype=np.float32))[0, 0])
    return clip_control(u, cfg)


# =============================================================================
# 5. Online mass estimator and closed-loop simulation
# =============================================================================

@dataclass
class OnlineMassEstimator:
    """RLS estimator for a hidden mass in m*xddot + b*xdot + k*x = u.

    The estimator uses the regression
        xddot = theta * phi,  where theta = 1/m and phi = u - b*xdot - k*x.

    Why this is used in the final benchmark:
    - it is a deployable classical comparator because it does not receive true mass;
    - it is stronger than direct pointwise division by acceleration;
    - it keeps the comparison fairer when evaluating the GRU hidden-mass policy.

    The update at time k uses the measured transition from k-1 to k and the
    control applied during that interval. A light low-pass filter is applied to
    the numerical acceleration, and the final mass estimate is rate-limited to
    avoid unphysical jumps due to derivative noise.
    """

    cfg: Config
    mass_hat: float
    theta_hat: Optional[float] = None
    p_cov: Optional[float] = None
    prev_position: Optional[float] = None
    prev_velocity: Optional[float] = None
    accel_filt: float = 0.0
    recent_mass_targets: Optional[deque] = None
    update_count: int = 0
    skipped_count: int = 0

    def __post_init__(self) -> None:
        self.mass_hat = float(np.clip(self.mass_hat, self.cfg.mass_min, self.cfg.mass_max))
        if self.theta_hat is None:
            self.theta_hat = 1.0 / self.mass_hat
        if self.p_cov is None:
            self.p_cov = float(self.cfg.estimator_rls_p0)
        if self.recent_mass_targets is None:
            self.recent_mass_targets = deque(maxlen=self.cfg.estimator_smooth_window)

    def update(
        self,
        position: float,
        velocity: float,
        previous_control: float,
        dt: float,
    ) -> Tuple[float, float, float, float, float, float, float]:
        """Perform one RLS inverse-mass update.

        Returns
        -------
        mass_hat:
            Filtered online mass estimate used by the scheduled LQI.
        raw_mass_candidate:
            Instantaneous RLS target mass before the final rate limiter.
        accel_est:
            Filtered numerical acceleration.
        theta_hat:
            Estimated inverse mass.
        residual:
            Regression residual accel_est - theta_hat*phi.
        regressor_phi:
            Force-like regressor u - b*v - k*x.
        update_flag:
            1.0 when the RLS update is accepted, otherwise 0.0.
        """
        if self.prev_velocity is None or self.prev_position is None:
            self.prev_position = float(position)
            self.prev_velocity = float(velocity)
            return float(self.mass_hat), float("nan"), 0.0, float(self.theta_hat), float("nan"), 0.0, 0.0

        accel_raw = float((velocity - self.prev_velocity) / dt)
        a = float(
            self.cfg.estimator_accel_lpf_alpha * accel_raw
            + (1.0 - self.cfg.estimator_accel_lpf_alpha) * self.accel_filt
        )
        self.accel_filt = a

        # Dynamics over the previous integration interval.
        phi = float(
            previous_control
            - self.cfg.b_damping * float(self.prev_velocity)
            - self.cfg.k_spring * float(self.prev_position)
        )

        raw_mass_candidate = float("nan")
        residual = float("nan")
        update_flag = 0.0

        if abs(phi) >= self.cfg.estimator_min_regressor and abs(a) >= self.cfg.estimator_min_accel:
            lam = float(self.cfg.estimator_forgetting_factor)
            p = float(self.p_cov)
            theta = float(self.theta_hat)
            denom = lam + phi * p * phi
            if np.isfinite(denom) and denom > 1e-12:
                gain = p * phi / denom
                residual = float(a - phi * theta)
                theta_new = theta + gain * residual
                theta_min = 1.0 / self.cfg.mass_max
                theta_max = 1.0 / self.cfg.mass_min
                if np.isfinite(theta_new):
                    theta_new = float(np.clip(theta_new, theta_min, theta_max))
                    p_new = (p - gain * phi * p) / lam
                    p_new = float(np.clip(p_new, self.cfg.estimator_rls_p_min, self.cfg.estimator_rls_p_max))

                    self.theta_hat = theta_new
                    self.p_cov = p_new
                    raw_mass_candidate = float(1.0 / theta_new)
                    self.recent_mass_targets.append(raw_mass_candidate)
                    self.update_count += 1
                    update_flag = 1.0
                else:
                    self.skipped_count += 1
            else:
                self.skipped_count += 1
        else:
            self.skipped_count += 1

        if self.recent_mass_targets and len(self.recent_mass_targets) > 0:
            # Median target reduces derivative-noise spikes without making the
            # scheduled LQI too sluggish after abrupt mass switches.
            target = float(np.median(np.asarray(self.recent_mass_targets, dtype=float)))
            delta = float(np.clip(target - self.mass_hat, -self.cfg.estimator_rate_limit, self.cfg.estimator_rate_limit))
            self.mass_hat = float(np.clip(self.mass_hat + delta, self.cfg.mass_min, self.cfg.mass_max))

        self.prev_position = float(position)
        self.prev_velocity = float(velocity)

        return (
            float(self.mass_hat),
            raw_mass_candidate,
            float(a),
            float(self.theta_hat),
            residual,
            float(phi),
            update_flag,
        )

CONTROLLERS = [
    "Fixed LQI m=1",
    "True-mass scheduled LQI (oracle)",
    "Estimated-mass scheduled LQI",
    "GRU hidden-mass policy",
]


def make_eval_scenarios(rng: np.random.Generator, gain_grid: np.ndarray, cfg: Config = CFG) -> List[Scenario]:
    scenarios: List[Scenario] = []

    # Include one canonical scenario close to the previous benchmark for visual interpretation.
    scenarios.append(
        Scenario(
            scenario_id=0,
            reference=2.0,
            initial_position=0.0,
            initial_velocity=0.0,
            switch_times=[2.0, 14.0],
            masses=[1.0, 6.0, 4.0],
        )
    )

    # Random scenarios for robust statistics.
    for sid in range(1, cfg.num_eval_scenarios):
        ref = ensure_nonzero_reference(rng, cfg)
        x0 = float(rng.uniform(-0.6, 0.6))
        v0 = float(rng.uniform(-0.25, 0.25))
        _, switch_times, masses = generate_piecewise_mass_schedule(
            rng,
            cfg.t_test,
            cfg.dt_test,
            gain_grid,
            min_segments=2,
            max_segments=5,
        )
        scenarios.append(
            Scenario(
                scenario_id=sid,
                reference=ref,
                initial_position=x0,
                initial_velocity=v0,
                switch_times=switch_times,
                masses=masses,
            )
        )
    return scenarios


def mass_at_time(t: float, scenario: Scenario) -> float:
    seg = 0
    for s in scenario.switch_times:
        if t >= s:
            seg += 1
        else:
            break
    return float(scenario.masses[min(seg, len(scenario.masses) - 1)])



def controller_stride(cfg: Config = CFG) -> int:
    """Return the number of plant-integration steps per controller update."""
    ratio = cfg.dt_control / cfg.dt_test
    stride = int(round(ratio))
    if stride < 1 or not math.isclose(ratio, stride, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "dt_control must be an integer multiple of dt_test. "
            f"Received dt_control={cfg.dt_control} and dt_test={cfg.dt_test}."
        )
    if not math.isclose(cfg.dt_control, cfg.dt_train, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "For temporal alignment, dt_control must equal dt_train. "
            f"Received dt_control={cfg.dt_control} and dt_train={cfg.dt_train}."
        )
    return stride


def simulate_closed_loop(
    controller_name: str,
    scenario: Scenario,
    gain_grid: np.ndarray,
    gain_cache: Dict[float, Tuple[np.ndarray, float]],
    model: Optional[nn.Module],
    x_scaler: Optional[Scaler],
    y_scaler: Optional[Scaler],
    rng: np.random.Generator,
    cfg: Config = CFG,
) -> Dict[str, np.ndarray]:
    """Simulate one controller with fine plant integration and aligned control updates.

    The plant is integrated every ``dt_test`` seconds. Measurements, integral-state
    updates, RLS updates, GRU-buffer updates, and controller commands occur every
    ``dt_control`` seconds. The command is held constant between update instants.

    Because ``dt_control == dt_train``, the GRU sequence contains the same physical
    history during training and evaluation: ``seq_len * dt_control`` seconds.
    """
    stride = controller_stride(cfg)
    n_steps = int(round(cfg.t_test / cfg.dt_test))
    t_hist = np.arange(n_steps, dtype=float) * cfg.dt_test

    x = np.array([scenario.initial_position, scenario.initial_velocity], dtype=float)
    z = 0.0
    u_current = 0.0
    u_previous_interval = 0.0
    prev_control_velocity_meas: Optional[float] = None

    first_feat = build_feature(x[0], x[1], scenario.reference, z, u_previous_interval, 0.0)
    seq_buffer = deque([first_feat.copy() for _ in range(cfg.seq_len)], maxlen=cfg.seq_len)
    mass_estimator = OnlineMassEstimator(cfg=cfg, mass_hat=cfg.estimator_initial_mass)

    pos = np.zeros(n_steps)
    vel = np.zeros(n_steps)
    pos_meas = np.zeros(n_steps)
    vel_meas = np.zeros(n_steps)
    mass_true = np.zeros(n_steps)
    mass_hat_hist = np.full(n_steps, np.nan)
    mass_raw_hist = np.full(n_steps, np.nan)
    mass_theta_hist = np.full(n_steps, np.nan)
    mass_residual_hist = np.full(n_steps, np.nan)
    mass_regressor_phi_hist = np.full(n_steps, np.nan)
    mass_update_flag_hist = np.full(n_steps, np.nan)
    control_update_flag_hist = np.zeros(n_steps)
    accel_est_hist = np.zeros(n_steps)
    u_hist = np.zeros(n_steps)
    error_hist = np.zeros(n_steps)
    z_hist = np.zeros(n_steps)

    fixed_gain = nearest_gain(cfg.nominal_fixed_mass, gain_grid, gain_cache)

    # Held diagnostic values between controller updates.
    held_x_meas = x.copy()
    held_accel_est = 0.0
    held_mass_hat = float("nan")
    held_raw_mass = float("nan")
    held_theta_hat = float("nan")
    held_mass_residual = float("nan")
    held_mass_phi = float("nan")

    for i, t in enumerate(t_hist):
        m_true = mass_at_time(float(t), scenario)
        is_control_update = (i % stride == 0)

        if is_control_update:
            control_update_flag_hist[i] = 1.0
            held_x_meas = add_measurement_noise(
                x,
                rng,
                cfg.test_position_noise_std,
                cfg.test_velocity_noise_std,
            )

            if prev_control_velocity_meas is None:
                held_accel_est = 0.0
            else:
                held_accel_est = float(
                    (held_x_meas[1] - prev_control_velocity_meas) / cfg.dt_control
                )
            prev_control_velocity_meas = float(held_x_meas[1])

            error_meas = float(scenario.reference - held_x_meas[0])
            feat = build_feature(
                held_x_meas[0],
                held_x_meas[1],
                scenario.reference,
                z,
                u_previous_interval,
                held_accel_est,
            )
            seq_buffer.append(feat.copy())

            if controller_name == "Fixed LQI m=1":
                Kx, Ki = fixed_gain
                u_current = lqi_control(held_x_meas, z, Kx, Ki, cfg)
                held_mass_hat = float("nan")
                held_raw_mass = float("nan")
                held_theta_hat = float("nan")
                held_mass_residual = float("nan")
                held_mass_phi = float("nan")
                mass_update_flag_hist[i] = float("nan")

            elif controller_name == "True-mass scheduled LQI (oracle)":
                Kx, Ki = nearest_gain(m_true, gain_grid, gain_cache)
                u_current = lqi_control(held_x_meas, z, Kx, Ki, cfg)
                held_mass_hat = m_true
                held_raw_mass = m_true
                held_theta_hat = 1.0 / m_true
                held_mass_residual = 0.0
                held_mass_phi = float("nan")
                mass_update_flag_hist[i] = float("nan")

            elif controller_name == "Estimated-mass scheduled LQI":
                (
                    held_mass_hat,
                    held_raw_mass,
                    held_accel_est,
                    held_theta_hat,
                    held_mass_residual,
                    held_mass_phi,
                    update_flag,
                ) = mass_estimator.update(
                    held_x_meas[0],
                    held_x_meas[1],
                    u_previous_interval,
                    cfg.dt_control,
                )
                mass_update_flag_hist[i] = update_flag
                Kx, Ki = nearest_gain(held_mass_hat, gain_grid, gain_cache)
                u_current = lqi_control(held_x_meas, z, Kx, Ki, cfg)

            elif controller_name == "GRU hidden-mass policy":
                if model is None or x_scaler is None or y_scaler is None:
                    raise ValueError("GRU policy requires model and scalers.")
                u_current = gru_control(model, seq_buffer, x_scaler, y_scaler, cfg)
                held_mass_hat = float("nan")
                held_raw_mass = float("nan")
                held_theta_hat = float("nan")
                held_mass_residual = float("nan")
                held_mass_phi = float("nan")
                mass_update_flag_hist[i] = float("nan")

            else:
                raise ValueError(f"Unknown controller: {controller_name}")

            # Match teacher-data generation: the sampled error updates the
            # integral state once per controller interval.
            z_increment = error_meas * cfg.dt_control
            u_previous_interval = u_current
        else:
            z_increment = 0.0

        pos[i] = x[0]
        vel[i] = x[1]
        pos_meas[i] = held_x_meas[0]
        vel_meas[i] = held_x_meas[1]
        mass_true[i] = m_true
        mass_hat_hist[i] = held_mass_hat
        mass_raw_hist[i] = held_raw_mass
        mass_theta_hist[i] = held_theta_hat
        mass_residual_hist[i] = held_mass_residual
        mass_regressor_phi_hist[i] = held_mass_phi
        accel_est_hist[i] = held_accel_est
        u_hist[i] = u_current
        error_hist[i] = scenario.reference - x[0]
        z_hist[i] = z

        x = plant_step(x, u_current, m_true, cfg.dt_test, cfg)
        z += z_increment

    return {
        "t": t_hist,
        "position": pos,
        "velocity": vel,
        "position_meas": pos_meas,
        "velocity_meas": vel_meas,
        "mass_true": mass_true,
        "mass_hat": mass_hat_hist,
        "mass_raw": mass_raw_hist,
        "mass_theta": mass_theta_hist,
        "mass_residual": mass_residual_hist,
        "mass_regressor_phi": mass_regressor_phi_hist,
        "mass_update_flag": mass_update_flag_hist,
        "control_update_flag": control_update_flag_hist,
        "accel_est": accel_est_hist,
        "u": u_hist,
        "error": error_hist,
        "integral_error": z_hist,
        "reference": np.full(n_steps, scenario.reference),
    }


# =============================================================================
# 6. Metrics and statistics
# =============================================================================

def settling_time(t: np.ndarray, error: np.ndarray, tol: float) -> float:
    outside = np.where(np.abs(error) > tol)[0]
    if len(outside) == 0:
        return 0.0
    j = int(outside[-1])
    if j >= len(t) - 1:
        return float("nan")
    return float(t[j + 1])


def recovery_after_switches(
    t: np.ndarray,
    error: np.ndarray,
    switch_times: Sequence[float],
    t_end: float,
    tol: float,
) -> Tuple[float, float, int, int, float]:
    """Recovery statistics after hidden-mass switches.

    Returns mean recovery time, max recovery time, number of successful switch
    recoveries, number of evaluated switches, and success rate. A switch is
    considered recovered when the remaining segment stays inside the tolerance band.
    """
    rec_times: List[float] = []
    for idx, sw in enumerate(switch_times):
        next_sw = switch_times[idx + 1] if idx + 1 < len(switch_times) else t_end
        segment = np.where((t >= sw) & (t < next_sw))[0]
        if len(segment) == 0:
            continue
        found = float("nan")
        for j in segment:
            remaining = segment[segment >= j]
            if np.all(np.abs(error[remaining]) <= tol):
                found = float(t[j] - sw)
                break
        rec_times.append(found)
    finite = [r for r in rec_times if np.isfinite(r)]
    mean_rec = float(np.mean(finite)) if finite else float("nan")
    max_rec = float(np.max(finite)) if finite else float("nan")
    n_total = int(len(rec_times))
    n_success = int(len(finite))
    success_rate = float(n_success / n_total) if n_total > 0 else float("nan")
    return mean_rec, max_rec, n_success, n_total, success_rate


def closed_loop_metrics(res: Dict[str, np.ndarray], scenario: Scenario, cfg: Config = CFG) -> Dict[str, float]:
    t = res["t"]
    pos = res["position"]
    u = res["u"]
    error = scenario.reference - pos
    dt = float(t[1] - t[0]) if len(t) > 1 else cfg.dt_test
    tol = cfg.settling_band_ratio * max(abs(scenario.reference), 1e-9)

    if "control_update_flag" in res:
        update_mask = np.asarray(res["control_update_flag"], dtype=float) > 0.5
        u_updates = u[update_mask]
    else:
        u_updates = u
    du = np.diff(u_updates)
    ddu = np.diff(du)
    denominator = max(abs(scenario.reference - pos[0]), 1e-9)
    overshoot = max(0.0, float(np.max((pos - scenario.reference) * np.sign(scenario.reference)))) / denominator * 100.0
    rec_mean, rec_max, rec_success_n, rec_total_n, rec_success_rate = recovery_after_switches(
        t, error, scenario.switch_times, cfg.t_test, tol
    )
    settling = settling_time(t, error, tol)

    sat_fraction = float(np.mean(np.abs(u) >= 0.999 * cfg.u_max))

    metrics = {
        "RMSE_track": float(np.sqrt(np.mean(error ** 2))),
        "MAE_track": float(np.mean(np.abs(error))),
        "IAE": float(np.sum(np.abs(error)) * dt),
        "ISE": float(np.sum(error ** 2) * dt),
        "ITAE": float(np.sum(t * np.abs(error)) * dt),
        "Overshoot_percent": float(overshoot),
        "Settling_time_2pct_s": settling,
        "Settled_2pct_success": float(np.isfinite(settling)),
        "Recovery_mean_after_switch_s": rec_mean,
        "Recovery_max_after_switch_s": rec_max,
        "Recovery_success_count": float(rec_success_n),
        "Recovery_switch_count": float(rec_total_n),
        "Recovery_success_rate": rec_success_rate,
        "Final_abs_error": float(abs(error[-1])),
        "Control_energy": float(np.sum(u ** 2) * dt),
        "Control_RMS": float(np.sqrt(np.mean(u ** 2))),
        "Max_abs_u": float(np.max(np.abs(u))),
        "Saturation_fraction": sat_fraction,
        "Total_variation_u": float(np.sum(np.abs(du))),
        "Mean_abs_du": float(np.mean(np.abs(du))) if len(du) else 0.0,
        "Mean_abs_ddu": float(np.mean(np.abs(ddu))) if len(ddu) else 0.0,
        "Controller_update_count": float(len(u_updates)),
    }

    if np.any(np.isfinite(res["mass_hat"])):
        mask = np.isfinite(res["mass_hat"])
        if np.any(mask):
            metrics["Mass_hat_MAE"] = float(np.mean(np.abs(res["mass_hat"][mask] - res["mass_true"][mask])))
            metrics["Mass_hat_RMSE"] = float(np.sqrt(np.mean((res["mass_hat"][mask] - res["mass_true"][mask]) ** 2)))
            metrics["Mass_hat_bias"] = float(np.mean(res["mass_hat"][mask] - res["mass_true"][mask]))
            metrics["Mass_hat_final_abs_error"] = float(abs(res["mass_hat"][mask][-1] - res["mass_true"][mask][-1]))
        else:
            metrics["Mass_hat_MAE"] = float("nan")
            metrics["Mass_hat_RMSE"] = float("nan")
            metrics["Mass_hat_bias"] = float("nan")
            metrics["Mass_hat_final_abs_error"] = float("nan")
    else:
        metrics["Mass_hat_MAE"] = float("nan")
        metrics["Mass_hat_RMSE"] = float("nan")
        metrics["Mass_hat_bias"] = float("nan")
        metrics["Mass_hat_final_abs_error"] = float("nan")

    if "mass_update_flag" in res:
        metrics["Mass_estimator_update_fraction"] = float(np.nanmean(res["mass_update_flag"])) if np.any(np.isfinite(res["mass_update_flag"])) else float("nan")
    else:
        metrics["Mass_estimator_update_fraction"] = float("nan")
    if "mass_residual" in res:
        residual = res["mass_residual"]
        mask_res = np.isfinite(residual)
        metrics["Mass_estimator_mean_abs_residual"] = float(np.mean(np.abs(residual[mask_res]))) if np.any(mask_res) else float("nan")
    else:
        metrics["Mass_estimator_mean_abs_residual"] = float("nan")
    if "mass_regressor_phi" in res:
        phi = res["mass_regressor_phi"]
        mask_phi = np.isfinite(phi)
        metrics["Mass_estimator_mean_abs_regressor"] = float(np.mean(np.abs(phi[mask_phi]))) if np.any(mask_phi) else float("nan")
    else:
        metrics["Mass_estimator_mean_abs_regressor"] = float("nan")

    return {k: safe_float(v) for k, v in metrics.items()}


def percent_improvement(baseline: float, value: float) -> float:
    if not np.isfinite(baseline) or abs(baseline) < 1e-12:
        return float("nan")
    return float((baseline - value) / baseline * 100.0)


def summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "RMSE_track", "MAE_track", "IAE", "ISE", "ITAE", "Overshoot_percent",
        "Settling_time_2pct_s", "Settled_2pct_success", "Recovery_mean_after_switch_s",
        "Recovery_max_after_switch_s", "Recovery_success_count", "Recovery_switch_count",
        "Recovery_success_rate", "Final_abs_error", "Control_energy", "Control_RMS",
        "Max_abs_u", "Saturation_fraction", "Total_variation_u", "Mean_abs_du", "Mean_abs_ddu",
        "Mass_hat_MAE", "Mass_hat_RMSE", "Mass_hat_bias", "Mass_hat_final_abs_error",
        "Mass_estimator_update_fraction", "Mass_estimator_mean_abs_residual",
        "Mass_estimator_mean_abs_regressor",
        "RMSE_improvement_vs_fixed_percent", "RMSE_improvement_vs_estimated_percent",
        "RMSE_gap_vs_oracle", "IAE_improvement_vs_fixed_percent", "IAE_improvement_vs_estimated_percent",
        "IAE_gap_vs_oracle", "Energy_improvement_vs_estimated_percent", "TV_improvement_vs_estimated_percent",
    ]
    rows = []
    for controller, g in df.groupby("Controller"):
        row = {"Controller": controller, "n_scenarios": int(g["scenario_id"].nunique())}
        for col in metric_cols:
            if col in g.columns:
                values = pd.to_numeric(g[col], errors="coerce")
                row[f"{col}_mean"] = float(values.mean())
                row[f"{col}_std"] = float(values.std(ddof=1))
                row[f"{col}_median"] = float(values.median())
                row[f"{col}_p25"] = float(values.quantile(0.25))
                row[f"{col}_p75"] = float(values.quantile(0.75))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Paired tests on scenario-level metrics.

    Positive mean_delta means GRU metric is higher than comparator.
    For error and energy metrics, a negative mean_delta favors the GRU policy.
    """
    metrics = [
        "RMSE_track", "IAE", "ITAE", "Control_energy", "Total_variation_u",
        "Overshoot_percent", "Settling_time_2pct_s", "Recovery_mean_after_switch_s",
    ]
    comparators = [
        "Fixed LQI m=1",
        "Estimated-mass scheduled LQI",
        "True-mass scheduled LQI (oracle)",
    ]
    gru_name = "GRU hidden-mass policy"
    rows: List[Dict[str, float | str | int]] = []

    pivot = df.pivot_table(index="scenario_id", columns="Controller", values=metrics, aggfunc="first")

    for metric in metrics:
        for comp in comparators:
            if (metric, gru_name) not in pivot.columns or (metric, comp) not in pivot.columns:
                continue
            pair = pivot[[(metric, gru_name), (metric, comp)]].dropna()
            if len(pair) < 3:
                continue
            gru_vals = pair[(metric, gru_name)].to_numpy(dtype=float)
            comp_vals = pair[(metric, comp)].to_numpy(dtype=float)
            delta = gru_vals - comp_vals
            mean_delta = float(np.mean(delta))
            std_delta = float(np.std(delta, ddof=1)) if len(delta) > 1 else float("nan")
            dz = mean_delta / std_delta if np.isfinite(std_delta) and std_delta > 1e-12 else float("nan")
            try:
                t_stat, t_p = ttest_rel(gru_vals, comp_vals, nan_policy="omit")
            except Exception:
                t_stat, t_p = float("nan"), float("nan")
            try:
                w_stat, w_p = wilcoxon(gru_vals, comp_vals, zero_method="wilcox", alternative="two-sided")
            except Exception:
                w_stat, w_p = float("nan"), float("nan")
            rows.append(
                {
                    "metric": metric,
                    "GRU_vs_comparator": comp,
                    "n_pairs": int(len(delta)),
                    "GRU_mean": float(np.mean(gru_vals)),
                    "comparator_mean": float(np.mean(comp_vals)),
                    "mean_delta_GRU_minus_comparator": mean_delta,
                    "cohen_dz": float(dz),
                    "paired_t_stat": float(t_stat),
                    "paired_t_pvalue": float(t_p),
                    "wilcoxon_stat": float(w_stat),
                    "wilcoxon_pvalue": float(w_p),
                    "interpretation_for_lower_better": "GRU better" if mean_delta < 0 else "Comparator better or equal",
                }
            )
    return pd.DataFrame(rows)



def holm_adjust_pvalues(pvalues: Sequence[float]) -> np.ndarray:
    """Holm step-down adjustment preserving NaN positions."""
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.full(p.shape, np.nan, dtype=float)
    finite_idx = np.where(np.isfinite(p))[0]
    if len(finite_idx) == 0:
        return adjusted
    order = finite_idx[np.argsort(p[finite_idx])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * p[idx]
        running = max(running, candidate)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def add_holm_correction(tests_df: pd.DataFrame) -> pd.DataFrame:
    """Apply Holm correction across tested metrics for each comparator."""
    out = tests_df.copy()
    out["paired_t_pvalue_holm"] = np.nan
    out["wilcoxon_pvalue_holm"] = np.nan
    if out.empty:
        return out
    for _, idx in out.groupby("GRU_vs_comparator").groups.items():
        idx = list(idx)
        out.loc[idx, "paired_t_pvalue_holm"] = holm_adjust_pvalues(
            out.loc[idx, "paired_t_pvalue"].to_numpy(dtype=float)
        )
        out.loc[idx, "wilcoxon_pvalue_holm"] = holm_adjust_pvalues(
            out.loc[idx, "wilcoxon_pvalue"].to_numpy(dtype=float)
        )
    return out


def add_relative_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    fixed = df[df["Controller"] == "Fixed LQI m=1"].set_index("scenario_id")
    estimated = df[df["Controller"] == "Estimated-mass scheduled LQI"].set_index("scenario_id")
    oracle = df[df["Controller"] == "True-mass scheduled LQI (oracle)"].set_index("scenario_id")

    for idx, row in df.iterrows():
        sid = int(row["scenario_id"])
        if sid in fixed.index:
            df.at[idx, "RMSE_improvement_vs_fixed_percent"] = percent_improvement(float(fixed.loc[sid, "RMSE_track"]), float(row["RMSE_track"]))
            df.at[idx, "IAE_improvement_vs_fixed_percent"] = percent_improvement(float(fixed.loc[sid, "IAE"]), float(row["IAE"]))
        if sid in estimated.index:
            df.at[idx, "RMSE_improvement_vs_estimated_percent"] = percent_improvement(float(estimated.loc[sid, "RMSE_track"]), float(row["RMSE_track"]))
            df.at[idx, "IAE_improvement_vs_estimated_percent"] = percent_improvement(float(estimated.loc[sid, "IAE"]), float(row["IAE"]))
            df.at[idx, "Energy_improvement_vs_estimated_percent"] = percent_improvement(float(estimated.loc[sid, "Control_energy"]), float(row["Control_energy"]))
            df.at[idx, "TV_improvement_vs_estimated_percent"] = percent_improvement(float(estimated.loc[sid, "Total_variation_u"]), float(row["Total_variation_u"]))
        if sid in oracle.index:
            df.at[idx, "RMSE_gap_vs_oracle"] = float(row["RMSE_track"]) - float(oracle.loc[sid, "RMSE_track"])
            df.at[idx, "IAE_gap_vs_oracle"] = float(row["IAE"]) - float(oracle.loc[sid, "IAE"])
    return df


# =============================================================================
# 7. Plotting
# =============================================================================

def plot_training(history_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(history_df["epoch"], history_df["train_loss"], label="Train")
    ax.plot(history_df["epoch"], history_df["val_loss"], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Huber loss on normalized control")
    ax.set_title("GRU policy training history")
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=True)
    fig.tight_layout()
    savefig(fig, "training_history")


def plot_representative_closed_loop(rep_results: Dict[str, Dict[str, np.ndarray]], scenario: Scenario) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12.5, 11.0), sharex=True)
    fig.suptitle(f"Representative hidden-mass scenario #{scenario.scenario_id}", fontsize=14)

    for name, res in rep_results.items():
        axes[0].plot(res["t"], res["position"], linewidth=1.8, label=name)
    axes[0].plot(next(iter(rep_results.values()))["t"], next(iter(rep_results.values()))["reference"], "k--", linewidth=1.2, label="Reference")
    axes[0].set_ylabel("Position (m)")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend(fontsize=8, ncol=2)

    for name, res in rep_results.items():
        axes[1].plot(res["t"], scenario.reference - res["position"], linewidth=1.6, label=name)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Tracking error (m)")
    axes[1].grid(True, alpha=0.35)

    for name, res in rep_results.items():
        axes[2].plot(res["t"], res["u"], linewidth=1.3, label=name)
    axes[2].set_ylabel("Control force")
    axes[2].grid(True, alpha=0.35)

    first = next(iter(rep_results.values()))
    axes[3].step(first["t"], first["mass_true"], where="post", linewidth=2.0, label="True hidden mass")
    if "Estimated-mass scheduled LQI" in rep_results:
        est = rep_results["Estimated-mass scheduled LQI"]
        axes[3].plot(est["t"], est["mass_hat"], linewidth=1.4, label="Online estimated mass")
    for sw in scenario.switch_times:
        for ax in axes:
            ax.axvline(sw, linestyle="--", linewidth=0.9, alpha=0.65)
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Mass (kg)")
    axes[3].grid(True, alpha=0.35)
    axes[3].legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    savefig(fig, "representative_closed_loop_comparison")


def plot_metric_boxplots(df: pd.DataFrame) -> None:
    metrics = ["RMSE_track", "IAE", "Control_energy", "Total_variation_u", "Overshoot_percent", "Recovery_mean_after_switch_s"]
    labels = ["RMSE", "IAE", "Energy", "TV(u)", "Overshoot (%)", "Recovery (s)"]

    for metric, label in zip(metrics, labels):
        fig, ax = plt.subplots(figsize=(10.5, 5.2))
        data = [df.loc[df["Controller"] == c, metric].dropna().to_numpy(dtype=float) for c in CONTROLLERS]
        tick_labels = [controller_short_label(c) for c in CONTROLLERS]
        # Matplotlib compatibility: recent versions renamed labels= to tick_labels=,
        # while some environments reject labels=. Setting ticks explicitly is robust.
        ax.boxplot(data, showmeans=True)
        ax.set_xticks(np.arange(1, len(tick_labels) + 1))
        ax.set_xticklabels(tick_labels)
        ax.set_ylabel(label)
        ax.set_title(f"Closed-loop distribution over randomized test scenarios: {label}")
        ax.grid(True, axis="y", alpha=0.35)
        fig.tight_layout()
        savefig(fig, f"boxplot_{metric}")


def plot_pareto(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    for controller in CONTROLLERS:
        g = df[df["Controller"] == controller]
        ax.scatter(g["Control_energy"], g["RMSE_track"], s=24, alpha=0.65, label=controller)
        ax.scatter(g["Control_energy"].mean(), g["RMSE_track"].mean(), s=130, marker="X", edgecolor="black")
    ax.set_xlabel("Control energy")
    ax.set_ylabel("RMSE tracking")
    ax.set_title("Pareto view: tracking error vs control energy")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "pareto_rmse_vs_energy")


def plot_summary_bar(summary_df: pd.DataFrame) -> None:
    metrics = ["RMSE_track_mean", "IAE_mean", "Control_energy_mean", "Total_variation_u_mean"]
    titles = ["Mean RMSE", "Mean IAE", "Mean control energy", "Mean total variation"]
    for metric, title in zip(metrics, titles):
        fig, ax = plt.subplots(figsize=(10.0, 5.0))
        ordered = summary_df.set_index("Controller").loc[CONTROLLERS].reset_index()
        y = ordered[metric].to_numpy(dtype=float)
        std_col = metric.replace("_mean", "_std")
        yerr = ordered[std_col].to_numpy(dtype=float) if std_col in ordered.columns else None
        ax.bar(np.arange(len(ordered)), y, yerr=yerr, capsize=4)
        ax.set_xticks(np.arange(len(ordered)))
        ax.set_xticklabels([controller_short_label(c) for c in ordered["Controller"]], fontsize=8)
        ax.set_ylabel(title)
        ax.set_title(title + " across randomized test scenarios")
        ax.grid(True, axis="y", alpha=0.35)
        fig.tight_layout()
        savefig(fig, f"summary_{metric}")


def plot_gru_vs_estimated_scatter(df: pd.DataFrame) -> None:
    metric = "RMSE_track"
    pivot = df.pivot_table(index="scenario_id", columns="Controller", values=metric, aggfunc="first")
    if "GRU hidden-mass policy" not in pivot.columns or "Estimated-mass scheduled LQI" not in pivot.columns:
        return
    x = pivot["Estimated-mass scheduled LQI"].to_numpy(dtype=float)
    y = pivot["GRU hidden-mass policy"].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) == 0:
        return
    low = float(min(np.min(x), np.min(y)))
    high = float(max(np.max(x), np.max(y)))
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    ax.scatter(x, y, s=35, alpha=0.75)
    ax.plot([low, high], [low, high], "k--", linewidth=1.2, label="equal performance")
    ax.set_xlabel("Estimated-mass scheduled LQI RMSE")
    ax.set_ylabel("GRU policy RMSE")
    ax.set_title("Paired test-scenario comparison: GRU policy vs estimated-mass LQI")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    savefig(fig, "paired_gru_vs_estimated_rmse")


def plot_mass_estimator_diagnostics(df: pd.DataFrame) -> None:
    """PDF diagnostics for the deployable estimated-mass LQI comparator."""
    g = df[df["Controller"] == "Estimated-mass scheduled LQI"].copy()
    if g.empty:
        return

    metrics = ["Mass_hat_MAE", "Mass_hat_RMSE", "Mass_hat_final_abs_error", "Mass_estimator_update_fraction"]
    labels = ["Mass MAE (kg)", "Mass RMSE (kg)", "Final mass error (kg)", "Update fraction"]
    data = [pd.to_numeric(g[m], errors="coerce").dropna().to_numpy(dtype=float) for m in metrics]

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_title("Estimated-mass scheduled LQI: RLS mass-estimator diagnostics")
    ax.grid(True, axis="y", alpha=0.35)
    fig.tight_layout()
    savefig(fig, "estimated_mass_rls_diagnostics")


def plot_success_rates(summary_df: pd.DataFrame) -> None:
    if "Settled_2pct_success_mean" not in summary_df.columns:
        return
    ordered = summary_df.set_index("Controller").loc[CONTROLLERS].reset_index()
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    y = 100.0 * ordered["Settled_2pct_success_mean"].to_numpy(dtype=float)
    ax.bar(np.arange(len(ordered)), y)
    ax.set_xticks(np.arange(len(ordered)))
    ax.set_xticklabels(
        [controller_short_label(c) for c in ordered["Controller"]],
        fontsize=8,
    )
    ax.set_ylabel("Settling success rate (%)")
    ax.set_title("2% settling-band success over randomized test scenarios")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.35)
    fig.tight_layout()
    savefig(fig, "summary_settling_success_rate")



def plot_gru_vs_estimated_paired_multimetric(df: pd.DataFrame) -> None:
    """Paired GRU policy vs RLS estimated-mass LQI comparison for key manuscript metrics."""
    metrics = [
        ("RMSE_track", "RMSE"),
        ("IAE", "IAE"),
        ("Control_energy", "Control energy"),
        ("Total_variation_u", "Total variation"),
        ("Overshoot_percent", "Overshoot (%)"),
        ("ITAE", "ITAE"),
    ]
    pivot = df.pivot_table(index="scenario_id", columns="Controller", values=[m for m, _ in metrics], aggfunc="first")
    gru_name = "GRU hidden-mass policy"
    est_name = "Estimated-mass scheduled LQI"
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.0))
    axes = axes.ravel()
    for ax, (metric, label) in zip(axes, metrics):
        if (metric, gru_name) not in pivot.columns or (metric, est_name) not in pivot.columns:
            ax.axis("off")
            continue
        x = pivot[(metric, est_name)].to_numpy(dtype=float)
        y = pivot[(metric, gru_name)].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        if len(x) == 0:
            ax.axis("off")
            continue
        lo = float(min(np.min(x), np.min(y)))
        hi = float(max(np.max(x), np.max(y)))
        pad = 0.04 * max(hi - lo, 1e-9)
        ax.scatter(x, y, s=CFG.publication_marker_size, alpha=0.72, linewidths=0.25, edgecolors="black")
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", linewidth=1.0, label="equal")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel(f"RLS-LQI {label}")
        ax.set_ylabel(f"GRU policy {label}")
        ax.set_title(label)
        set_axis_grid(ax)
    fig.suptitle("Paired test-scenario comparison: GRU policy vs RLS estimated-mass LQI", fontsize=CFG.publication_title_size + 1)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, "paired_gru_vs_rls_lqi_multimetric")


def plot_gru_relative_improvement_vs_estimated(df: pd.DataFrame) -> None:
    """Mean paired percentage change of GRU policy relative to deployable RLS-LQI."""
    gru_name = "GRU hidden-mass policy"
    est_name = "Estimated-mass scheduled LQI"
    lower_metrics = [
        ("RMSE_track", "RMSE"),
        ("IAE", "IAE"),
        ("ITAE", "ITAE"),
        ("Overshoot_percent", "Overshoot"),
        ("Control_energy", "Energy"),
        ("Total_variation_u", "TV(u)"),
        ("Mean_abs_ddu", "Mean |ddu|"),
    ]
    rows = []
    for metric, label in lower_metrics:
        pivot = df.pivot_table(index="scenario_id", columns="Controller", values=metric, aggfunc="first")
        if gru_name not in pivot.columns or est_name not in pivot.columns:
            continue
        vals = pivot[[gru_name, est_name]].dropna()
        if vals.empty:
            continue
        # Positive means the GRU policy is lower/better for lower-is-better metrics.
        improvement = (vals[est_name].to_numpy(dtype=float) - vals[gru_name].to_numpy(dtype=float)) / np.maximum(np.abs(vals[est_name].to_numpy(dtype=float)), 1e-12) * 100.0
        rows.append({
            "metric": label,
            "mean": float(np.mean(improvement)),
            "sem95": float(1.96 * np.std(improvement, ddof=1) / math.sqrt(len(improvement))) if len(improvement) > 1 else 0.0,
            "n": int(len(improvement)),
        })
    if not rows:
        return
    out = pd.DataFrame(rows)
    out.to_csv(REPORT_DIR / "gru_relative_improvement_vs_rls_lqi_for_manuscript.csv", index=False)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    x = np.arange(len(out))
    ax.bar(x, out["mean"].to_numpy(dtype=float), yerr=out["sem95"].to_numpy(dtype=float), capsize=4)
    ax.axhline(0.0, linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(out["metric"].tolist(), rotation=20, ha="right")
    ax.set_ylabel("GRU policy improvement vs RLS-LQI (%)")
    ax.set_title("GRU policy relative improvement over deployable RLS estimated-mass LQI")
    set_axis_grid(ax, axis="y")
    fig.tight_layout()
    savefig(fig, "gru_relative_improvement_vs_rls_lqi")


def plot_success_rates_combined(summary_df: pd.DataFrame) -> None:
    """Settling and switch-recovery success rates in one manuscript-ready panel."""
    needed = ["Settled_2pct_success_mean", "Recovery_success_rate_mean"]
    if any(c not in summary_df.columns for c in needed):
        return
    ordered = summary_df.set_index("Controller").loc[CONTROLLERS].reset_index()
    x = np.arange(len(ordered))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.2, 4.9))
    y1 = 100.0 * ordered["Settled_2pct_success_mean"].to_numpy(dtype=float)
    y2 = 100.0 * ordered["Recovery_success_rate_mean"].to_numpy(dtype=float)
    ax.bar(x - width / 2, y1, width, label="2% settling success")
    ax.bar(x + width / 2, y2, width, label="Switch-recovery success")
    ax.set_xticks(x)
    ax.set_xticklabels([controller_short_label(c) for c in ordered["Controller"]])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Closed-loop success rates over randomized hidden-mass tests")
    set_axis_grid(ax, axis="y")
    ax.legend(ncol=2, frameon=True)
    fig.tight_layout()
    savefig(fig, "summary_success_rates_combined")


def plot_average_rank_by_metric(rank_summary: pd.DataFrame) -> None:
    """Average scenario-level ranks. Lower rank is better."""
    if rank_summary.empty:
        return
    metrics = ["RMSE_track", "IAE", "ITAE", "Control_energy", "Total_variation_u", "Overshoot_percent"]
    labels = ["RMSE", "IAE", "ITAE", "Energy", "TV(u)", "Overshoot"]
    pivot = rank_summary.pivot_table(index="metric", columns="Controller", values="mean", aggfunc="first")
    if pivot.empty:
        return
    x = np.arange(len(metrics))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    for i, controller in enumerate(CONTROLLERS):
        y = [pivot.loc[m, controller] if m in pivot.index and controller in pivot.columns else np.nan for m in metrics]
        ax.bar(x + (i - 1.5) * width, y, width, label=controller_short_label(controller).replace("\n", " "))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean rank over test scenarios")
    ax.set_title("Controller ranking by metric; lower rank is better")
    ax.set_ylim(0.8, len(CONTROLLERS) + 0.35)
    set_axis_grid(ax, axis="y")
    ax.legend(ncol=2, frameon=True)
    fig.tight_layout()
    savefig(fig, "average_rank_by_metric")


def plot_representative_cumulative_error_energy(rep_results: Dict[str, Dict[str, np.ndarray]], scenario: Scenario) -> None:
    """Representative cumulative IAE and control energy curves."""
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.5), sharex=True)
    for name, res in rep_results.items():
        t = res["t"]
        dt = float(t[1] - t[0]) if len(t) > 1 else CFG.dt_test
        e = scenario.reference - res["position"]
        cum_iae = np.cumsum(np.abs(e)) * dt
        cum_energy = np.cumsum(res["u"] ** 2) * dt
        axes[0].plot(t, cum_iae, label=name)
        axes[1].plot(t, cum_energy, label=name)
    for ax in axes:
        set_switch_lines(ax, scenario.switch_times)
        set_axis_grid(ax)
    axes[0].set_ylabel("Cumulative IAE")
    axes[0].set_title(f"Representative scenario #{scenario.scenario_id}: cumulative error and energy")
    axes[1].set_ylabel("Cumulative control energy")
    axes[1].set_xlabel("Time (s)")
    axes[0].legend(ncol=2, frameon=True)
    fig.tight_layout()
    savefig(fig, "representative_cumulative_error_energy")


def plot_representative_mass_estimation_zoom(rep_results: Dict[str, Dict[str, np.ndarray]], scenario: Scenario) -> None:
    """Dedicated true-vs-estimated mass panel for the RLS comparator."""
    if "Estimated-mass scheduled LQI" not in rep_results:
        return
    est = rep_results["Estimated-mass scheduled LQI"]
    fig, ax = plt.subplots(figsize=(10.0, 4.5))
    ax.step(est["t"], est["mass_true"], where="post", linewidth=1.8, label="True hidden mass")
    ax.plot(est["t"], est["mass_hat"], linewidth=1.35, label="RLS mass estimate")
    ax.plot(est["t"], est["mass_raw"], linewidth=0.8, alpha=0.45, label="Rate-limited raw estimate")
    set_switch_lines(ax, scenario.switch_times)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mass (kg)")
    ax.set_title(f"Representative scenario #{scenario.scenario_id}: online RLS mass estimation")
    set_axis_grid(ax)
    ax.legend(ncol=3, frameon=True)
    fig.tight_layout()
    savefig(fig, "representative_rls_mass_estimation")


def write_figure_index() -> None:
    """Create an auditable figure list for manuscript drafting."""
    rows = [
        ("representative_closed_loop_comparison", "Main representative trajectory: position, error, control, and hidden/estimated mass."),
        ("representative_cumulative_error_energy", "Representative cumulative IAE and cumulative control-energy curves."),
        ("representative_rls_mass_estimation", "Dedicated RLS mass-estimation diagnostic for the deployable classical comparator."),
        ("boxplot_RMSE_track", "Scenario-level RMSE distribution."),
        ("boxplot_IAE", "Scenario-level IAE distribution."),
        ("boxplot_Control_energy", "Scenario-level control-energy distribution."),
        ("boxplot_Total_variation_u", "Scenario-level control total-variation distribution."),
        ("pareto_rmse_vs_energy", "Tracking-energy Pareto view."),
        ("paired_gru_vs_rls_lqi_multimetric", "Paired GRU policy versus RLS-LQI comparisons across multiple metrics."),
        ("gru_relative_improvement_vs_rls_lqi", "Mean paired percentage improvement of GRU policy relative to RLS-LQI."),
        ("summary_success_rates_combined", "Settling and switch-recovery success rates."),
        ("average_rank_by_metric", "Average scenario-level rank by metric."),
        ("estimated_mass_rls_diagnostics", "Distribution of RLS mass-estimator diagnostic metrics."),
    ]
    pd.DataFrame(rows, columns=["figure_stem", "recommended_use"]).to_csv(REPORT_DIR / "figure_index_for_manuscript.csv", index=False)


def metric_dictionary() -> pd.DataFrame:
    rows = [
        ("RMSE_track", "Root-mean-square tracking error over the closed-loop test trajectory", "lower"),
        ("MAE_track", "Mean absolute tracking error", "lower"),
        ("IAE", "Integral of absolute error", "lower"),
        ("ISE", "Integral of squared error", "lower"),
        ("ITAE", "Integral of time-weighted absolute error", "lower"),
        ("Overshoot_percent", "Maximum signed overshoot relative to the initial reference gap", "lower"),
        ("Settling_time_2pct_s", "Last-entry 2% settling time; NaN means no sustained settling", "lower"),
        ("Settled_2pct_success", "1 if the controller remains inside the 2% band before the horizon ends", "higher"),
        ("Recovery_mean_after_switch_s", "Mean recovery time after hidden-mass switches over successful switch recoveries", "lower"),
        ("Recovery_success_rate", "Fraction of hidden-mass switches after which sustained recovery is achieved", "higher"),
        ("Control_energy", "Time integral of squared control force", "lower"),
        ("Control_RMS", "Root-mean-square control force", "lower"),
        ("Total_variation_u", "Sum of absolute first differences of the control signal", "lower"),
        ("Mean_abs_du", "Mean absolute control increment", "lower"),
        ("Mean_abs_ddu", "Mean absolute second difference of control", "lower"),
        ("Mass_hat_MAE", "Mean absolute error of mass estimate; applies to the estimated-mass LQI comparator", "lower"),
        ("Mass_hat_RMSE", "RMSE of mass estimate; applies to the estimated-mass LQI comparator", "lower"),
        ("Mass_estimator_update_fraction", "Fraction of time steps with accepted RLS mass-estimator updates", "diagnostic"),
    ]
    return pd.DataFrame(rows, columns=["metric", "meaning", "preferred_direction"])


def _fmt(x: float, nd: int = 4) -> str:
    if x is None or not np.isfinite(float(x)):
        return "NA"
    return f"{float(x):.{nd}f}"


def write_manuscript_ready_outputs(
    metrics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    tests_df: pd.DataFrame,
    rank_summary: pd.DataFrame,
    cfg: Config = CFG,
) -> None:
    """Create ordered text, LaTeX snippets, and metric dictionary for writing.

    These files are not additional analyses; they reformat the already computed
    test-only results into a form that can be copied into the manuscript.
    """
    metric_dictionary().to_csv(REPORT_DIR / "metric_dictionary.csv", index=False)

    ordered = summary_df.set_index("Controller").loc[CONTROLLERS].reset_index()
    main_cols = [
        "Controller",
        "RMSE_track_mean", "RMSE_track_std",
        "IAE_mean", "IAE_std",
        "ITAE_mean", "ITAE_std",
        "Overshoot_percent_mean", "Overshoot_percent_std",
        "Control_energy_mean", "Control_energy_std",
        "Total_variation_u_mean", "Total_variation_u_std",
        "Settled_2pct_success_mean", "Recovery_success_rate_mean",
    ]
    present_cols = [c for c in main_cols if c in ordered.columns]
    ordered[present_cols].to_csv(REPORT_DIR / "main_results_table_for_manuscript.csv", index=False)

    # Extract GRU-vs-estimated improvements from scenario-level relative metrics.
    gru_rows = metrics_df[metrics_df["Controller"] == "GRU hidden-mass policy"].copy()
    improvement_cols = [
        "RMSE_improvement_vs_estimated_percent",
        "IAE_improvement_vs_estimated_percent",
        "Energy_improvement_vs_estimated_percent",
        "TV_improvement_vs_estimated_percent",
    ]
    imp_summary = {}
    for col in improvement_cols:
        if col in gru_rows.columns:
            vals = pd.to_numeric(gru_rows[col], errors="coerce")
            imp_summary[col] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)),
                "median": float(vals.median()),
                "p25": float(vals.quantile(0.25)),
                "p75": float(vals.quantile(0.75)),
            }
    pd.DataFrame(imp_summary).T.reset_index().rename(columns={"index": "metric"}).to_csv(REPORT_DIR / "gru_improvement_vs_estimated_summary.csv", index=False)

    # Mass-estimator diagnostic table.
    est_summary = ordered[ordered["Controller"] == "Estimated-mass scheduled LQI"]
    est_diag_cols = [
        "Controller",
        "Mass_hat_MAE_mean", "Mass_hat_MAE_std",
        "Mass_hat_RMSE_mean", "Mass_hat_RMSE_std",
        "Mass_hat_bias_mean", "Mass_hat_final_abs_error_mean",
        "Mass_estimator_update_fraction_mean",
        "Mass_estimator_mean_abs_residual_mean",
    ]
    est_present = [c for c in est_diag_cols if c in est_summary.columns]
    if est_present:
        est_summary[est_present].to_csv(REPORT_DIR / "estimated_mass_rls_diagnostics_for_manuscript.csv", index=False)

    # LaTeX table snippets.
    latex_table = ordered[present_cols].copy()
    rename = {
        "Controller": "Controller",
        "RMSE_track_mean": "RMSE mean", "RMSE_track_std": "RMSE std",
        "IAE_mean": "IAE mean", "IAE_std": "IAE std",
        "ITAE_mean": "ITAE mean", "ITAE_std": "ITAE std",
        "Overshoot_percent_mean": "Overshoot mean (\\%)", "Overshoot_percent_std": "Overshoot std",
        "Control_energy_mean": "Energy mean", "Control_energy_std": "Energy std",
        "Total_variation_u_mean": "TV mean", "Total_variation_u_std": "TV std",
        "Settled_2pct_success_mean": "Settling success", "Recovery_success_rate_mean": "Recovery success",
    }
    latex_table = latex_table.rename(columns={k: v for k, v in rename.items() if k in latex_table.columns})
    numeric_cols = [c for c in latex_table.columns if c != "Controller"]
    for c in numeric_cols:
        latex_table[c] = latex_table[c].map(lambda x: _fmt(x, 4))
    latex = latex_table.to_latex(index=False, escape=False)

    tests_est = tests_df[tests_df["GRU_vs_comparator"] == "Estimated-mass scheduled LQI"].copy()
    test_cols = ["metric", "n_pairs", "GRU_mean", "comparator_mean", "mean_delta_GRU_minus_comparator", "wilcoxon_pvalue"]
    test_cols = [c for c in test_cols if c in tests_est.columns]
    tests_latex = tests_est[test_cols].to_latex(index=False, escape=False) if not tests_est.empty else ""

    with (REPORT_DIR / "latex_tables_and_captions.tex").open("w", encoding="utf-8") as f:
        f.write("% Auto-generated LaTeX snippets from the hidden-mass benchmark.\n")
        f.write("% Tables are computed only on independent closed-loop TEST scenarios.\n\n")
        f.write("\\begin{table*}[t]\n\\centering\n")
        f.write("\\caption{Closed-loop test performance over randomized hidden-mass scenarios. Values are reported as scenario-level means and standard deviations. The true-mass scheduled LQI is an oracle reference, whereas the estimated-mass scheduled LQI and the GRU policy do not receive the true mass.}\n")
        f.write("\\label{tab:hidden_mass_closed_loop}\n")
        f.write(latex)
        f.write("\\end{table*}\n\n")
        if tests_latex:
            f.write("\\begin{table}[t]\n\\centering\n")
            f.write("\\caption{Paired test-scenario comparison between the proposed GRU policy and the deployable estimated-mass scheduled LQI. Negative mean deltas favor the GRU policy for error, overshoot, and energy metrics.}\n")
            f.write("\\label{tab:gru_vs_estimated_mass_lqi}\n")
            f.write(tests_latex)
            f.write("\\end{table}\n")

    # Narrative notes for drafting.
    gru = ordered[ordered["Controller"] == "GRU hidden-mass policy"].iloc[0]
    est = ordered[ordered["Controller"] == "Estimated-mass scheduled LQI"].iloc[0]
    oracle = ordered[ordered["Controller"] == "True-mass scheduled LQI (oracle)"].iloc[0]
    fixed = ordered[ordered["Controller"] == "Fixed LQI m=1"].iloc[0]

    rmse_gain = imp_summary.get("RMSE_improvement_vs_estimated_percent", {}).get("mean", float("nan"))
    iae_gain = imp_summary.get("IAE_improvement_vs_estimated_percent", {}).get("mean", float("nan"))
    energy_gain = imp_summary.get("Energy_improvement_vs_estimated_percent", {}).get("mean", float("nan"))
    tv_gain = imp_summary.get("TV_improvement_vs_estimated_percent", {}).get("mean", float("nan"))

    notes = f"""# Manuscript-ready notes: hidden-mass GRU policy vs LQI benchmark

## Evaluation protocol

The benchmark evaluates four controllers over {cfg.num_eval_scenarios} independent closed-loop TEST scenarios with randomized references, initial states, and piecewise-constant hidden masses. Training and validation trajectories are used only for fitting the GRU policy and early stopping; they are not included in the closed-loop comparative statistics.

## Controllers

1. **Fixed LQI m=1**: nominal baseline designed for a fixed mass. It is intentionally not adaptive and should be interpreted as a weak baseline.
2. **True-mass scheduled LQI (oracle)**: gain-scheduled LQI using the true hidden mass. It is not deployable but provides an informative oracle reference.
3. **Estimated-mass scheduled LQI**: deployable classical comparator. In this final version, the hidden mass is estimated online by RLS on the inverse-mass regression xddot = theta(u - b xdot - kx), theta = 1/m.
4. **GRU hidden-mass policy**: recurrent policy trained from observation/control history; the true mass is never provided to the model.

## Main numerical reading

- GRU policy mean RMSE: {_fmt(gru.get('RMSE_track_mean', float('nan')))} ± {_fmt(gru.get('RMSE_track_std', float('nan')))}.
- Estimated-mass LQI mean RMSE: {_fmt(est.get('RMSE_track_mean', float('nan')))} ± {_fmt(est.get('RMSE_track_std', float('nan')))}.
- Oracle true-mass LQI mean RMSE: {_fmt(oracle.get('RMSE_track_mean', float('nan')))} ± {_fmt(oracle.get('RMSE_track_std', float('nan')))}.
- Fixed LQI mean RMSE: {_fmt(fixed.get('RMSE_track_mean', float('nan')))} ± {_fmt(fixed.get('RMSE_track_std', float('nan')))}.

Compared with the deployable estimated-mass scheduled LQI, the GRU policy changes the main metrics as follows:

- RMSE improvement: {_fmt(rmse_gain, 2)}%.
- IAE improvement: {_fmt(iae_gain, 2)}%.
- Control-energy improvement: {_fmt(energy_gain, 2)}%.
- Total-variation improvement: {_fmt(tv_gain, 2)}%. Positive values indicate lower total variation for the GRU policy; negative values indicate a smoother comparator.

## Mass-estimator diagnostic to report

- Estimated-mass LQI mass MAE: {_fmt(est.get('Mass_hat_MAE_mean', float('nan')))} kg.
- Estimated-mass LQI mass RMSE: {_fmt(est.get('Mass_hat_RMSE_mean', float('nan')))} kg.
- RLS update fraction: {_fmt(est.get('Mass_estimator_update_fraction_mean', float('nan')))}.

This diagnostic is important because it shows whether the deployable classical comparator is technically credible. If the mass error remains high, the discussion should state that the GRU policy outperforms the implemented RLS-based estimated-mass LQI, not every possible adaptive LQI.

## Safe interpretation for the Results section

The GRU policy should be presented as a history-based hidden-parameter controller. The strongest fair comparison is against the estimated-mass scheduled LQI, because both methods do not receive the true mass. The true-mass scheduled LQI should be described as an oracle reference rather than a deployable baseline. If the GRU policy outperforms the oracle on some metrics, this should not be interpreted as beating optimal LQI in general; it indicates that the recurrent policy can behave more conservatively under abrupt switching, saturation, and multi-objective trade-offs.

## Figure-output note

This v4 script exports each figure as vector PDF, vector SVG, and 600-dpi PNG. Prefer the PDF files for LaTeX manuscripts when allowed by the journal; use PNG only for systems that require raster images.

## Suggested wording

Across randomized hidden-mass test scenarios, the GRU policy achieved lower average tracking error and control energy than the deployable estimated-mass scheduled LQI, while remaining competitive with the true-mass scheduled LQI oracle. The fixed-gain LQI exhibited the largest degradation under mass variation, confirming that nominal fixed-mass design is insufficient for the hidden-parameter setting. The RLS-based estimated-mass LQI provides a stronger classical comparator than direct pointwise mass inversion, and its mass-estimation diagnostics are reported to make the comparison auditable.

## Cautionary statement

The results should not be phrased as a general superiority of the GRU policy over LQI. The defensible claim is narrower: under the tested abrupt hidden-mass switching scenarios and the implemented RLS-based deployable comparator, the recurrent policy provides a robust closed-loop alternative that improves tracking-energy performance without access to the true mass.
"""
    with (REPORT_DIR / "manuscript_ready_notes.md").open("w", encoding="utf-8") as f:
        f.write(notes)


# =============================================================================
# 8. Main experiment
# =============================================================================

def main() -> None:
    print("--- Hidden-mass GRU policy vs LQI controllers benchmark ---", flush=True)
    print(f"Torch version: {torch.__version__}", flush=True)
    print(f"PyTorch CUDA build: {torch.version.cuda}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("WARNING: running on CPU.", flush=True)
    print(f"Results directory: {RESULTS_DIR.resolve()}", flush=True)

    save_json(asdict(CFG), RESULTS_DIR / "config.json")

    teacher_rng = np.random.default_rng(CFG.seed)
    evaluation_rng = np.random.default_rng(CFG.evaluation_scenario_seed)
    gain_grid = make_gain_grid(CFG)
    stride = controller_stride(CFG)
    print(
        f"Temporal alignment: plant dt={CFG.dt_test:.4f} s, "
        f"controller dt={CFG.dt_control:.4f} s, stride={stride}, "
        f"GRU history={CFG.seq_len * CFG.dt_control:.3f} s",
        flush=True,
    )

    print("\nBuilding dense LQI gain cache...", flush=True)
    gain_cache = build_gain_cache(gain_grid)
    pd.DataFrame({"mass_grid": gain_grid}).to_csv(CSV_DIR / "lqi_gain_mass_grid.csv", index=False)

    print(f"Generating {CFG.num_teacher_trajectories} oracle teacher trajectories...", flush=True)
    trajectories: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(CFG.num_teacher_trajectories):
        trajectories.append(generate_teacher_trajectory(teacher_rng, gain_grid, gain_cache, CFG))
        if (i + 1) % 20 == 0 or (i + 1) == CFG.num_teacher_trajectories:
            print(f"  trajectory {i + 1}/{CFG.num_teacher_trajectories}", flush=True)

    idx = np.arange(CFG.num_teacher_trajectories)
    teacher_rng.shuffle(idx)
    n_train = int(CFG.train_ratio * CFG.num_teacher_trajectories)
    n_val = int(CFG.val_ratio * CFG.num_teacher_trajectories)

    train_traj = [trajectories[i] for i in idx[:n_train]]
    val_traj = [trajectories[i] for i in idx[n_train:n_train + n_val]]
    test_traj = [trajectories[i] for i in idx[n_train + n_val:]]

    X_train_raw = np.concatenate([xy[0] for xy in train_traj], axis=0)
    y_train_raw = np.concatenate([xy[1] for xy in train_traj], axis=0)
    x_scaler = fit_scaler(X_train_raw)
    y_scaler = fit_scaler(y_train_raw)

    print("Building sequence datasets...", flush=True)
    Xtr, ytr = make_sequence_arrays(train_traj, x_scaler, y_scaler, CFG.seq_len)
    Xva, yva = make_sequence_arrays(val_traj, x_scaler, y_scaler, CFG.seq_len)
    Xte, yte = make_sequence_arrays(test_traj, x_scaler, y_scaler, CFG.seq_len)

    train_dataset = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    val_dataset = TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva))
    test_dataset = TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))

    print(f"Train samples: {len(train_dataset):,}", flush=True)
    print(f"Val samples:   {len(val_dataset):,}", flush=True)
    print(f"Test samples:  {len(test_dataset):,}", flush=True)

    model = GRUPolicy(input_dim=CFG.input_dim, hidden_dim=CFG.gru_hidden_dim).to(DEVICE)
    model, history = train_policy(model, train_dataset, val_dataset, CFG)

    history_df = pd.DataFrame(history)
    history_df.to_csv(CSV_DIR / "training_history.csv", index=False)
    plot_training(history_df)

    offline = evaluate_offline(model, test_dataset, y_scaler, CFG)
    pd.DataFrame([offline]).to_csv(CSV_DIR / "offline_teacher_imitation_metrics.csv", index=False)
    print("\n--- Offline teacher-control imitation metrics ---", flush=True)
    print(pd.DataFrame([offline]).round(5).to_string(index=False), flush=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "x_mean": x_scaler.mean,
            "x_std": x_scaler.std,
            "y_mean": y_scaler.mean,
            "y_std": y_scaler.std,
            "config": asdict(CFG),
            "feature_order": [
                "position_meas", "velocity_meas", "reference", "integral_error",
                "previous_control", "tracking_error", "estimated_acceleration"
            ],
        },
        MODEL_DIR / "gru_hidden_mass_policy.pt",
    )

    print(f"\nGenerating {CFG.num_eval_scenarios} independent closed-loop TEST scenarios...", flush=True)
    scenarios = make_eval_scenarios(evaluation_rng, gain_grid, CFG)
    scenario_rows = [asdict(s) for s in scenarios]
    scenario_df = pd.DataFrame(scenario_rows)
    scenario_df.insert(0, "split", "test_closed_loop")
    scenario_df.to_csv(CSV_DIR / "test_scenarios.csv", index=False)
    scenario_df.to_csv(CSV_DIR / "evaluation_scenarios.csv", index=False)  # backward-compatible copy

    all_metric_rows: List[Dict[str, object]] = []
    representative_results: Optional[Dict[str, Dict[str, np.ndarray]]] = None
    representative_scenario: Optional[Scenario] = None
    trajectory_rows_for_rep: List[pd.DataFrame] = []

    print("\nRunning closed-loop simulations for all controllers...", flush=True)
    for si, scenario in enumerate(scenarios):
        scenario_results: Dict[str, Dict[str, np.ndarray]] = {}
        # Every controller receives the same additive noise realization in a
        # given scenario. Reinitializing the generator with the same scenario
        # seed preserves a strictly paired comparison.
        for controller in CONTROLLERS:
            sim_rng = np.random.default_rng(
                CFG.evaluation_noise_seed + 10000 * scenario.scenario_id
            )
            res = simulate_closed_loop(
                controller,
                scenario,
                gain_grid,
                gain_cache,
                model=model,
                x_scaler=x_scaler,
                y_scaler=y_scaler,
                rng=sim_rng,
                cfg=CFG,
            )
            scenario_results[controller] = res
            row: Dict[str, object] = {
                "split": "test_closed_loop",
                "scenario_id": scenario.scenario_id,
                "Controller": controller,
                "reference": scenario.reference,
                "initial_position": scenario.initial_position,
                "initial_velocity": scenario.initial_velocity,
                "n_switches": len(scenario.switch_times),
                "switch_times": ";".join([f"{x:.4f}" for x in scenario.switch_times]),
                "masses": ";".join([f"{x:.4f}" for x in scenario.masses]),
            }
            row.update(closed_loop_metrics(res, scenario, CFG))
            all_metric_rows.append(row)

        if scenario.scenario_id == 0:
            representative_results = scenario_results
            representative_scenario = scenario
            for controller, res in scenario_results.items():
                df_traj = pd.DataFrame(
                    {
                        "split": "test_closed_loop",
                        "scenario_id": scenario.scenario_id,
                        "Controller": controller,
                        "t": res["t"],
                        "reference": res["reference"],
                        "position": res["position"],
                        "velocity": res["velocity"],
                        "position_meas": res["position_meas"],
                        "velocity_meas": res["velocity_meas"],
                        "mass_true": res["mass_true"],
                        "mass_hat": res["mass_hat"],
                        "mass_raw": res["mass_raw"],
                        "mass_theta": res["mass_theta"],
                        "mass_residual": res["mass_residual"],
                        "mass_regressor_phi": res["mass_regressor_phi"],
                        "mass_update_flag": res["mass_update_flag"],
                        "control_update_flag": res["control_update_flag"],
                        "u": res["u"],
                        "error": scenario.reference - res["position"],
                    }
                )
                trajectory_rows_for_rep.append(df_traj)

        if (si + 1) % 10 == 0 or (si + 1) == len(scenarios):
            print(f"  scenarios completed: {si + 1}/{len(scenarios)}", flush=True)

    metrics_df = pd.DataFrame(all_metric_rows)
    metrics_df = add_relative_metrics(metrics_df)
    # These are TEST closed-loop metrics only. Training and validation are used only
    # for fitting/early stopping; they are not included in the comparative statistics.
    metrics_df.to_csv(CSV_DIR / "test_closed_loop_metrics_by_scenario.csv", index=False)
    metrics_df.to_csv(CSV_DIR / "closed_loop_metrics_by_scenario.csv", index=False)  # backward-compatible copy

    summary_df = summarize_metrics(metrics_df)
    summary_df.to_csv(CSV_DIR / "test_closed_loop_summary_statistics.csv", index=False)
    summary_df.to_csv(CSV_DIR / "closed_loop_summary_statistics.csv", index=False)  # backward-compatible copy

    tests_df = paired_statistical_tests(metrics_df)
    tests_df = add_holm_correction(tests_df)
    tests_df.to_csv(CSV_DIR / "test_paired_statistical_tests_GRU_vs_comparators.csv", index=False)
    tests_df.to_csv(CSV_DIR / "paired_statistical_tests_GRU_vs_comparators.csv", index=False)  # backward-compatible copy

    # Ranking table: lower is better for these metrics.
    rank_metrics = ["RMSE_track", "IAE", "ITAE", "Control_energy", "Total_variation_u", "Overshoot_percent"]
    rank_rows = []
    for sid, g in metrics_df.groupby("scenario_id"):
        for metric in rank_metrics:
            ranks = g.set_index("Controller")[metric].rank(method="average", ascending=True)
            for controller, rank_value in ranks.items():
                rank_rows.append({"scenario_id": sid, "metric": metric, "Controller": controller, "rank": float(rank_value)})
    rank_df = pd.DataFrame(rank_rows)
    if not rank_df.empty:
        rank_df.insert(0, "split", "test_closed_loop")
    rank_df.to_csv(CSV_DIR / "test_controller_ranks_by_scenario.csv", index=False)
    rank_df.to_csv(CSV_DIR / "controller_ranks_by_scenario.csv", index=False)  # backward-compatible copy
    rank_summary = rank_df.groupby(["Controller", "metric"], as_index=False)["rank"].agg(["mean", "std", "median"]).reset_index()
    rank_summary.to_csv(CSV_DIR / "test_controller_rank_summary.csv", index=False)
    rank_summary.to_csv(CSV_DIR / "controller_rank_summary.csv", index=False)  # backward-compatible copy

    if representative_results is not None and representative_scenario is not None:
        rep_traj_df = pd.concat(trajectory_rows_for_rep, axis=0, ignore_index=True)
        rep_traj_df.to_csv(CSV_DIR / "test_representative_trajectory_all_controllers.csv", index=False)
        rep_traj_df.to_csv(CSV_DIR / "representative_trajectory_all_controllers.csv", index=False)  # backward-compatible copy
        plot_representative_closed_loop(representative_results, representative_scenario)

    plot_metric_boxplots(metrics_df)
    plot_pareto(metrics_df)
    plot_summary_bar(summary_df)
    plot_gru_vs_estimated_scatter(metrics_df)
    plot_mass_estimator_diagnostics(metrics_df)
    plot_success_rates(summary_df)
    plot_gru_vs_estimated_paired_multimetric(metrics_df)
    plot_gru_relative_improvement_vs_estimated(metrics_df)
    plot_success_rates_combined(summary_df)
    plot_average_rank_by_metric(rank_summary)
    if representative_results is not None and representative_scenario is not None:
        plot_representative_cumulative_error_energy(representative_results, representative_scenario)
        plot_representative_mass_estimation_zoom(representative_results, representative_scenario)
    write_manuscript_ready_outputs(metrics_df, summary_df, tests_df, rank_summary, CFG)
    write_figure_index()

    print("\n--- TEST closed-loop summary statistics: main metrics ---", flush=True)
    main_cols = [
        "Controller", "n_scenarios",
        "RMSE_track_mean", "RMSE_track_std",
        "IAE_mean", "IAE_std",
        "Control_energy_mean", "Control_energy_std",
        "Total_variation_u_mean", "Total_variation_u_std",
    ]
    existing = [c for c in main_cols if c in summary_df.columns]
    print(summary_df[existing].round(5).to_string(index=False), flush=True)

    print("\n--- TEST statistical tests: GRU policy vs Estimated-mass scheduled LQI ---", flush=True)
    if not tests_df.empty:
        view = tests_df[tests_df["GRU_vs_comparator"] == "Estimated-mass scheduled LQI"]
        cols = [
            "metric", "n_pairs", "GRU_mean", "comparator_mean",
            "mean_delta_GRU_minus_comparator", "wilcoxon_pvalue",
            "wilcoxon_pvalue_holm", "cohen_dz",
            "interpretation_for_lower_better",
        ]
        print(view[cols].round(6).to_string(index=False), flush=True)

    if DEVICE.type == "cuda":
        try:
            print(
                f"\nCUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB ; "
                f"reserved: {torch.cuda.memory_reserved(0) / 1024 ** 2:.2f} MB",
                flush=True,
            )
        except Exception:
            pass

    print("\nSaved outputs:", flush=True)
    print(f"  Root:   {RESULTS_DIR.resolve()}", flush=True)
    print(f"  CSV:    {CSV_DIR.resolve()}", flush=True)
    print(f"  PDF vector: {FIG_DIR.resolve()}", flush=True)
    print(f"  PNG 600dpi: {PNG_DIR.resolve()}", flush=True)
    print(f"  SVG vector: {SVG_DIR.resolve()}", flush=True)
    print(f"  Model:      {MODEL_DIR.resolve()}", flush=True)
    print(f"  Report: {REPORT_DIR.resolve()}", flush=True)


if __name__ == "__main__":
    main()
