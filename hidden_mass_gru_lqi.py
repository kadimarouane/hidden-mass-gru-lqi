"""
GRU-based hidden-mass control benchmark
=======================================

This script compares four controllers on a hidden-mass
mass--spring--damper tracking problem:

1) Fixed LQI designed at a nominal mass
2) True-mass gain-scheduled LQI used as an oracle reference
3) RLS estimated-mass gain-scheduled LQI
4) GRU hidden-mass policy trained from observation/control history

Design principles
-----------------
- The dataset split is performed at the trajectory level to avoid leakage.
- The GRU policy receives only measured history and previous control actions.
- The true mass is not provided to the GRU policy or to the RLS-based controller.
- The true-mass scheduled LQI is used only as an oracle reference.
- Final metrics are computed on independent randomized closed-loop test scenarios.
- Outputs include metrics, statistical summaries, trained models, and vector figures.

Run
---
python hidden_mass_gru_lqi_benchmark.py

Required packages
-----------------
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
    """Configuration for the hidden-mass GRU/LQI benchmark."""

    # Reproducibility
    seed: int = 42

    # Output directory
    output_root: str = "results"
    experiment_name: str = "gru_hidden_mass_lqi_benchmark"

    # Device selection
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

    # Teacher trajectory generation
    dt_train: float = 0.02
    t_train: float = 10.0
    num_teacher_trajectories: int = 144
    train_ratio: float = 0.70
    val_ratio: float = 0.15

    # GRU policy training
    seq_len: int = 30
    batch_size: int = 512
    max_epochs: int = 35
    early_stop_patience: int = 7
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    huber_beta: float = 0.5

    # Measurement noise
    train_position_noise_std: float = 0.002
    train_velocity_noise_std: float = 0.003
    test_position_noise_std: float = 0.002
    test_velocity_noise_std: float = 0.003

    # Closed-loop evaluation
    dt_test: float = 0.01
    t_test: float = 21.0
    num_eval_scenarios: int = 60
    settling_band_ratio: float = 0.02
    reference_min_abs: float = 0.8
    reference_range_low: float = -2.5
    reference_range_high: float = 3.0

    # RLS inverse-mass estimator
    estimator_initial_mass: float = 1.0
    estimator_forgetting_factor: float = 0.992
    estimator_rls_p0: float = 20.0
    estimator_rls_p_min: float = 1e-4
    estimator_rls_p_max: float = 1e4
    estimator_min_regressor: float = 0.08
    estimator_min_accel: float = 0.03
    estimator_accel_lpf_alpha: float = 0.22
    estimator_rate_limit: float = 0.12
    estimator_smooth_window: int = 9

    # Figure export
    figure_dpi: int = 600
    save_png: bool = True
    save_svg: bool = True
    font_size: int = 9
    title_size: int = 10
    label_size: int = 9
    legend_size: int = 8
    line_width: float = 1.45
    marker_size: float = 28.0

    # GRU input features:
    # position, velocity, reference, integral error,
    # previous control, tracking error, estimated acceleration.
    # The true mass is intentionally excluded.
    input_dim: int = 7
    gru_hidden_dim: int = 72


CFG = Config()


# =============================================================================
# 1. Plot configuration
# =============================================================================

def configure_matplotlib(cfg: Config = CFG) -> None:
    """Configure Matplotlib for vector and high-resolution raster exports."""
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": cfg.figure_dpi,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.035,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "DejaVu Sans",
        "font.size": cfg.font_size,
        "axes.titlesize": cfg.title_size,
        "axes.labelsize": cfg.label_size,
        "xtick.labelsize": cfg.font_size,
        "ytick.labelsize": cfg.font_size,
        "legend.fontsize": cfg.legend_size,
        "axes.linewidth": 0.75,
        "lines.linewidth": cfg.line_width,
        "lines.markersize": 4.2,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "grid.linewidth": 0.45,
        "path.simplify": True,
        "path.simplify_threshold": 0.01,
        "agg.path.chunksize": 20000,
    })


configure_matplotlib(CFG)


# =============================================================================
# 2. Controller names and output folders
# =============================================================================

CONTROLLER_FIXED_LQI = "Fixed LQI (nominal mass)"
CONTROLLER_ORACLE_LQI = "True-mass scheduled LQI (oracle)"
CONTROLLER_RLS_LQI = "RLS estimated-mass scheduled LQI"
CONTROLLER_GRU = "GRU hidden-mass policy"

CONTROLLERS = [
    CONTROLLER_FIXED_LQI,
    CONTROLLER_ORACLE_LQI,
    CONTROLLER_RLS_LQI,
    CONTROLLER_GRU,
]


def controller_short_label(name: str) -> str:
    """Return compact controller labels for figures."""
    labels = {
        CONTROLLER_FIXED_LQI: "Fixed LQI\nnominal mass",
        CONTROLLER_ORACLE_LQI: "True-mass\nLQI oracle",
        CONTROLLER_RLS_LQI: "RLS estimated-mass\nLQI",
        CONTROLLER_GRU: "GRU\nhidden-mass policy",
    }
    return labels.get(name, name)


def select_device(cfg: Config) -> torch.device:
    """Select CUDA when available, otherwise CPU."""
    if cfg.prefer_cuda and torch.cuda.is_available():
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        return torch.device("cuda:0")

    if cfg.require_cuda:
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot access a CUDA device. "
            "Check the NVIDIA driver and the installed PyTorch build."
        )

    return torch.device("cpu")


DEVICE = select_device(CFG)
PIN_MEMORY = DEVICE.type == "cuda"

RESULTS_DIR = Path(CFG.output_root) / CFG.experiment_name
FIG_DIR = RESULTS_DIR / "figures" / "pdf"
SVG_DIR = RESULTS_DIR / "figures" / "svg"
PNG_DIR = RESULTS_DIR / "figures" / "png"
CSV_DIR = RESULTS_DIR / "metrics"
MODEL_DIR = RESULTS_DIR / "models"
REPORT_DIR = RESULTS_DIR / "reports"

for directory in [RESULTS_DIR, FIG_DIR, SVG_DIR, PNG_DIR, CSV_DIR, MODEL_DIR, REPORT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def savefig(fig: plt.Figure, stem: str, cfg: Config = CFG) -> None:
    """Save a figure as PDF, SVG, and optionally PNG."""
    metadata = {
        "Creator": "hidden_mass_gru_lqi_benchmark.py",
        "Title": stem,
    }

    fig.align_labels()

    fig.savefig(
        FIG_DIR / f"{stem}.pdf",
        format="pdf",
        dpi=cfg.figure_dpi,
        bbox_inches="tight",
        pad_inches=0.035,
        metadata=metadata,
    )

    if cfg.save_svg:
        fig.savefig(
            SVG_DIR / f"{stem}.svg",
            format="svg",
            dpi=cfg.figure_dpi,
            bbox_inches="tight",
            pad_inches=0.035,
            metadata=metadata,
        )

    if cfg.save_png:
        fig.savefig(
            PNG_DIR / f"{stem}.png",
            format="png",
            dpi=cfg.figure_dpi,
            bbox_inches="tight",
            pad_inches=0.035,
        )

    plt.close(fig)


# =============================================================================
# 3. Compatibility aliases for replacing old labels in the remaining script
# =============================================================================

# Use these constants throughout the rest of the benchmark instead of hard-coded
# strings. This avoids informal terminology and makes later refactoring safer.

LABEL_REPLACEMENTS = {
    "Fixed LQI m=1": CONTROLLER_FIXED_LQI,
    "True-mass scheduled LQI (oracle)": CONTROLLER_ORACLE_LQI,
    "Estimated-mass scheduled LQI": CONTROLLER_RLS_LQI,
    "AI-GRU hidden-mass policy": CONTROLLER_GRU,
}
