"""Matplotlib rendering of the Gate G1 calibration figures (paper-ready).

Consumes arrays + binned results from :mod:`plot.eval.calibration` (no metric math here). Agg
backend so it needs neither a display nor ffmpeg.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from plot.eval import calibration as cal  # noqa: E402


def reliability_diagram(
    epv: np.ndarray, y: np.ndarray, weight: np.ndarray | None, out_path: str | Path,
    *, n_bins: int = 15, slope: dict | None = None, title: str = "EPV calibration (G1, held-out games)",
) -> None:
    """Reliability diagram for the continuous EPV forecast + an EPV histogram subplot."""
    bins = cal.reliability_bins(epv, y, weight, n_bins=n_bins, strategy="quantile")
    px = [b["pred_mean"] for b in bins]
    oy = [b["obs_mean"] for b in bins]
    lim = [0, max(max(px), max(oy), 3.0) * 1.02]

    fig, (ax, axh) = plt.subplots(2, 1, figsize=(6.4, 7.0), height_ratios=[3, 1], sharex=True)
    ax.plot(lim, lim, "--", color="#888", lw=1, label="perfect")
    ax.plot(px, oy, "o-", color="#1d428a", lw=1.6, ms=5, label="EPV (quantile bins)")
    if slope and np.isfinite(slope.get("slope", float("nan"))):
        xs = np.array(lim)
        ax.plot(xs, slope["intercept"] + slope["slope"] * xs, ":", color="#c8102e", lw=1.2,
                label=f"fit: y={slope['slope']:.2f}x{slope['intercept']:+.2f}")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_ylabel("mean realized points")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)

    axh.hist(np.asarray(epv), bins=40, color="#1d428a", alpha=0.7)
    axh.set_xlabel("predicted EPV (expected points)")
    axh.set_ylabel("frames")
    axh.grid(alpha=0.25)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def per_class_reliability(probs: np.ndarray, y: np.ndarray, weight: np.ndarray | None,
                          out_path: str | Path, *, n_bins: int = 10) -> None:
    """One-vs-rest reliability curve for each points class {0..K-1}."""
    K = probs.shape[1]
    res = cal.per_class_reliability(probs, y, weight, n_bins=n_bins)
    fig, axes = plt.subplots(1, K, figsize=(3.0 * K, 3.0), sharex=True, sharey=True)
    for k in range(K):
        ax = axes[k]
        b = res["classwise_bins"][k]
        ax.plot([0, 1], [0, 1], "--", color="#888", lw=1)
        if b:
            ax.plot([x["pred_mean"] for x in b], [x["obs_mean"] for x in b], "o-", color="#007a33", ms=4)
        ax.set_title(f"P(points={k})", fontsize=9)
        ax.set_xlabel("predicted")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("empirical frequency")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(f"Per-class reliability (macro ECE={res['macro_ece']:.3f})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
