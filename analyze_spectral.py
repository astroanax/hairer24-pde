#!/usr/bin/env python3
"""Spectral analysis plots from sweep results."""

import numpy as np, os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.abspath(__file__))

sweep_file = os.path.join(base_dir, "sweep_results.csv")
if os.path.exists(sweep_file):
    data = np.genfromtxt(sweep_file, delimiter=",", names=True)
    kappas = data["kappa"]
    lyaps = data["lyapunov"]
    batchelors = data["batchelor_scale"]
    spectral_medians = data["spectral_median"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.loglog(kappas, np.abs(lyaps), "bo-", linewidth=2, markersize=8, label="|lambda|")
    k_fine = np.logspace(np.log10(kappas.min() * 0.5), np.log10(kappas.max() * 2), 50)
    C_fit = np.abs(lyaps[2]) * kappas[2] ** 4
    ax.loglog(
        k_fine, C_fit * k_fine ** (-4), "r--", alpha=0.5, label=r"$\propto \kappa^{-4}$"
    )
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$|\lambda|$")
    ax.set_title(r"Theorem 2.1: $|\lambda| \leq C \kappa^{-q}$")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    valid = batchelors > 0
    ax.loglog(kappas[valid], batchelors[valid], "go-", linewidth=2, markersize=8)
    k_fine = np.logspace(
        np.log10(kappas[valid].min() * 0.5), np.log10(kappas[valid].max() * 2), 50
    )
    c_fit = batchelors[valid][2] / np.sqrt(kappas[valid][2])
    ax.loglog(
        k_fine,
        c_fit * np.sqrt(k_fine),
        "g--",
        alpha=0.5,
        label=r"$\propto \sqrt{\kappa}$",
    )
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$\chi_B$")
    ax.set_title("Batchelor Scale")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.loglog(kappas, spectral_medians, "ms-", linewidth=2, markersize=8)
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$k_M$")
    ax.set_title("Spectral Median")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(base_dir, "scaling_analysis.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()

spec_file = os.path.join(base_dir, "output", "spectrum.csv")
if os.path.exists(spec_file):
    spec = np.genfromtxt(spec_file, delimiter=",", names=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    valid = spec["energy"] > 0
    ax.loglog(spec["wavenumber"][valid], spec["energy"][valid], "b-", linewidth=2)
    ax.set_xlabel(r"$|k|$")
    ax.set_ylabel(r"$E(k)$")
    ax.set_title("Scalar Energy Spectrum")
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    energy = spec["energy"]
    k = spec["wavenumber"]
    total = np.sum(energy)
    if total > 0:
        cumsum = np.cumsum(energy) / total
        ax.plot(k, cumsum, "b-", linewidth=2)
        ax.axhline(y=0.5, color="r", linestyle="--", alpha=0.5, label="50%")
        median_idx = np.searchsorted(cumsum, 0.5)
        if median_idx < len(k):
            ax.axvline(
                x=k[median_idx],
                color="m",
                linestyle="--",
                alpha=0.5,
                label=f"$k_M$ = {k[median_idx]:.3f}",
            )
        ax.set_xlabel(r"$|k|$")
        ax.set_ylabel("Cumulative energy")
        ax.set_title("Spectral Median")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(base_dir, "spectral_distribution.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

ts_file = os.path.join(base_dir, "output", "timeseries.csv")
meta_file = os.path.join(base_dir, "output", "metadata.txt")
if os.path.exists(ts_file) and os.path.exists(meta_file):
    ts = np.genfromtxt(ts_file, delimiter=",", names=True)
    meta = {}
    with open(meta_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                meta[parts[0]] = parts[1]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes[0, 0].plot(ts["time"], ts["lyapunov"], "b-", linewidth=1)
    if "lyap" in meta:
        axes[0, 0].axhline(
            y=float(meta["lyap"]),
            color="r",
            linestyle="--",
            label=f"Final: {float(meta['lyap']):.4f}",
        )
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel(r"$\lambda$")
    axes[0, 0].set_title("Lyapunov Convergence")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(ts["time"], ts["energy"], "r-", linewidth=1)
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("Energy")
    axes[0, 1].set_title("Energy")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 2].plot(ts["time"], ts["enstrophy"], "g-", linewidth=1)
    axes[0, 2].set_xlabel("Time")
    axes[0, 2].set_ylabel("Enstrophy")
    axes[0, 2].set_title("Enstrophy")
    axes[0, 2].grid(True, alpha=0.3)
    axes[1, 0].plot(ts["time"], ts["scalar_norm"], "m-", linewidth=1)
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel(r"$||\rho||$")
    axes[1, 0].set_title("Scalar Decay")
    axes[1, 0].set_yscale("log")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].axis("off")
    info = f"Grid: {meta.get('Nx', '?')}x{meta.get('Ny', '?')}\nnu={meta.get('nu', '?')} kappa={meta.get('kappa', '?')}\nsigma={meta.get('sigma', '?')}\n\n"
    info += f"lambda={meta.get('lyap', '?')}\nchi_B={meta.get('batchelor_scale', '?')}\nk_B={meta.get('batchelor_k', '?')}\nk_M={meta.get('spectral_median', '?')}"
    axes[1, 1].text(
        0.1,
        0.9,
        info,
        transform=axes[1, 1].transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    if len(ts) > 1:
        dt_vals = np.diff(ts["time"])
        log_ratios = np.diff(np.log(ts["scalar_norm"] + 1e-30)) / dt_vals
        axes[1, 2].plot(ts["time"][1:], log_ratios, "c-", linewidth=0.5, alpha=0.7)
        axes[1, 2].axhline(
            y=float(meta.get("lyap", 0)),
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"avg: {float(meta.get('lyap', 0)):.4f}",
        )
        axes[1, 2].set_xlabel("Time")
        axes[1, 2].set_ylabel("d(log|rho|)/dt")
        axes[1, 2].set_title("Instantaneous Growth Rate")
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(base_dir, "detailed_analysis.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
