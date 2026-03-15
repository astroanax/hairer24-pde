# Hairer et al. (2024) - Lyapunov Exponent for SNS

CUDA implementation of the simulation framework from:

> Hairer, Punshon-Smith, Rosati, Yi (2024)
> *Lower bounds on the top Lyapunov exponent for linear PDEs driven by the 2D stochastic Navier-Stokes equations*
> [arXiv:2411.10419](https://arxiv.org/abs/2411.10419)

Solves the vorticity form of the 2D stochastic Navier-Stokes equations, advects a passive scalar, and computes the top Lyapunov exponent of the tangent equation via Benettin renormalization. Also computes the spectral median and Batchelor scale at each output step.

## Requirements

- NVIDIA GPU with CUDA (developed on RTX 4050, sm_89)
- CUDA toolkit with cuFFT and cuRAND
- Python 3 with numpy, matplotlib, pandas
- ffmpeg (for video output)

## Build

```bash
chmod +x build.sh
./build.sh
```

## Run

Single simulation:
```bash
./solver --Nx 64 --Ny 64 --nu 0.005 --kappa 0.005 --sigma 3.0 \
         --dt 0.0005 --T 30 --si 200 --lr 100 --outdir output
```

Kappa sweep (Theorem 2.1 scaling):
```bash
python3 run_sweep.py
```

## Visualize

```bash
python3 visualize.py          # panel PNGs, GIF, MP4 from output/
python3 analyze_spectral.py   # scaling plots from sweep_results.csv
```

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--Nx`, `--Ny` | 64 | Grid resolution |
| `--nu` | 0.005 | Fluid viscosity |
| `--kappa` | 0.005 | Scalar diffusivity |
| `--sigma` | 2.0 | Noise amplitude |
| `--dt` | 0.0005 | Time step |
| `--T` | 25 | Total simulation time |
| `--si` | 200 | Save interval (steps) |
| `--lr` | 100 | Renormalization interval (steps) |

## Output

- `timeseries.csv` - time, energy, enstrophy, Lyapunov estimate, scalar norm
- `metadata.txt` - grid params, final lambda, Batchelor scale, spectral median
- `spectrum.csv` - shell-averaged scalar energy spectrum
- `frame_*.bin` - binary snapshots (vorticity, scalar, tangent, velocity)

## Key results

Theorem 2.1: λ ≥ -C κ^{-q} for any q > 3. The code verifies this lower bound holds numerically even as κ → 0.

The Batchelor scale χ_B = √(κ/|λ|) gives the characteristic dissipation length. The spectral median k_M (wavenumber where half the scalar energy sits above/below) tracks how fine the scalar structures get over time.
