#!/usr/bin/env python3
"""Kappa sweep to verify Theorem 2.1 scaling."""

import subprocess, os
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
kappas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
sigma, nu = 3.0, 0.005
Nx, Ny = 32, 32
dt = 0.0005
T_warmup, T_run = 5.0, 15.0

results = []
for kappa in kappas:
    outpath = os.path.join(base_dir, f"sweep_kappa_{kappa:.4f}")
    os.makedirs(outpath, exist_ok=True)
    cmd = [
        os.path.join(base_dir, "solver"),
        "--Nx",
        str(Nx),
        "--Ny",
        str(Ny),
        "--nu",
        str(nu),
        "--kappa",
        str(kappa),
        "--sigma",
        str(sigma),
        "--dt",
        str(dt),
        "--T",
        str(T_warmup + T_run),
        "--si",
        "200",
        "--lr",
        "100",
        "--outdir",
        outpath,
    ]
    print(f"kappa={kappa}...", end=" ", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAIL: {result.stderr[:100]}")
        continue

    meta = {}
    with open(os.path.join(outpath, "metadata.txt")) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                meta[parts[0]] = parts[1]

    lyap = float(meta.get("lyap", "nan"))
    batchelor = float(meta.get("batchelor_scale", "nan"))
    smedian = float(meta.get("spectral_median", "nan"))
    bk = float(meta.get("batchelor_k", "nan"))
    results.append(
        {
            "kappa": kappa,
            "lyap": lyap,
            "batchelor": batchelor,
            "spectral_median": smedian,
            "batchelor_k": bk,
        }
    )
    print(f"lambda={lyap:.6f} chi_B={batchelor:.4e} k_M={smedian:.4f}")

with open(os.path.join(base_dir, "sweep_results.csv"), "w") as f:
    f.write("kappa,lyapunov,batchelor_scale,spectral_median,batchelor_k\n")
    for r in results:
        f.write(
            f"{r['kappa']},{r['lyap']},{r['batchelor']},{r['spectral_median']},{r['batchelor_k']}\n"
        )

print(f"\n{'kappa':>10} {'lambda':>12} {'chi_B':>12} {'k_M':>10} {'k_B':>10}")
print("-" * 60)
for r in results:
    print(
        f"{r['kappa']:10.4f} {r['lyap']:12.6f} {r['batchelor']:12.4e} {r['spectral_median']:10.4f} {r['batchelor_k']:10.4f}"
    )
