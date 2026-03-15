#!/usr/bin/env python3
"""Creates panel PNGs, GIF, MP4, and analysis plots from solver binary output."""

import numpy as np, struct, glob, os, subprocess
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def read_frame(filename):
    with open(filename, "rb") as f:
        Nx, Ny = struct.unpack("ii", f.read(8))
        n = Nx * Ny
        omega = np.frombuffer(f.read(8 * n), dtype=np.float64).reshape(Nx, Ny)
        scalar = np.frombuffer(f.read(8 * n), dtype=np.float64).reshape(Nx, Ny)
        tangent = np.frombuffer(f.read(8 * n), dtype=np.float64).reshape(Nx, Ny)
        u1 = np.frombuffer(f.read(8 * n), dtype=np.float64).reshape(Nx, Ny)
        u2 = np.frombuffer(f.read(8 * n), dtype=np.float64).reshape(Nx, Ny)
    return omega, scalar, tangent, u1, u2


outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
frames = sorted(glob.glob(os.path.join(outdir, "frame_*.bin")))
print(f"{len(frames)} frames")

ts = pd.read_csv(os.path.join(outdir, "timeseries.csv"))
meta = {}
with open(os.path.join(outdir, "metadata.txt")) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            meta[parts[0]] = parts[1]
Nx, Ny = int(meta["Nx"]), int(meta["Ny"])
x = np.linspace(0, 2 * np.pi, Nx, endpoint=False)
y = np.linspace(0, 2 * np.pi, Ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing="ij")

panel_dir = os.path.join(outdir, "panels")
os.makedirs(panel_dir, exist_ok=True)

for i, fname in enumerate(frames):
    omega, scalar, tangent, u1, u2 = read_frame(fname)
    speed = np.sqrt(u1**2 + u2**2)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    vmax = max(abs(omega.min()), abs(omega.max()), 0.01)
    axes[0, 0].pcolormesh(
        X, Y, omega, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto"
    )
    axes[0, 0].set_title(f"Vorticity (t={ts['time'].iloc[i]:.2f})")
    axes[0, 0].set_aspect("equal")
    vmax_s = max(abs(scalar.min()), abs(scalar.max()), 0.001)
    axes[0, 1].pcolormesh(
        X, Y, scalar, cmap="RdBu_r", vmin=-vmax_s, vmax=vmax_s, shading="auto"
    )
    axes[0, 1].set_title("Scalar")
    axes[0, 1].set_aspect("equal")
    skip = max(1, Nx // 16)
    axes[1, 0].pcolormesh(X, Y, speed, cmap="hot", shading="auto")
    axes[1, 0].quiver(
        X[::skip, ::skip],
        Y[::skip, ::skip],
        u1[::skip, ::skip],
        u2[::skip, ::skip],
        color="cyan",
        scale=20,
    )
    axes[1, 0].set_title("Velocity")
    axes[1, 0].set_aspect("equal")
    vmax_t = max(abs(tangent.min()), abs(tangent.max()), 0.001)
    axes[1, 1].pcolormesh(
        X, Y, tangent, cmap="RdBu_r", vmin=-vmax_t, vmax=vmax_t, shading="auto"
    )
    axes[1, 1].set_title(f"Tangent (lyap={ts['lyapunov'].iloc[i]:.3f})")
    axes[1, 1].set_aspect("equal")
    plt.tight_layout()
    plt.savefig(
        os.path.join(panel_dir, f"panel_{i:05d}.png"), dpi=80, bbox_inches="tight"
    )
    plt.close()
    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(frames)}")

subprocess.run(
    [
        "ffmpeg",
        "-y",
        "-framerate",
        "10",
        "-i",
        os.path.join(panel_dir, "panel_%05d.png"),
        "-vf",
        "scale=640:640:flags=lanczos",
        "-gifflags",
        "+transdiff",
        os.path.join(outdir, "simulation.gif"),
    ],
    capture_output=True,
)
subprocess.run(
    [
        "ffmpeg",
        "-y",
        "-framerate",
        "10",
        "-i",
        os.path.join(panel_dir, "panel_%05d.png"),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
        os.path.join(outdir, "simulation.mp4"),
    ],
    capture_output=True,
)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].plot(ts["time"], ts["lyapunov"], "b-")
axes[0, 0].set_title("Lyapunov")
axes[0, 0].grid(True, alpha=0.3)
axes[0, 1].plot(ts["time"], ts["energy"], "r-")
axes[0, 1].set_title("Energy")
axes[0, 1].grid(True, alpha=0.3)
axes[1, 0].plot(ts["time"], ts["enstrophy"], "g-")
axes[1, 0].set_title("Enstrophy")
axes[1, 0].grid(True, alpha=0.3)
axes[1, 1].plot(ts["time"], ts["scalar_norm"], "m-")
axes[1, 1].set_title("Scalar")
axes[1, 1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(outdir, "analysis.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Done")
