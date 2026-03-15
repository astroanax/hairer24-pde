# Lyapunov Exponent for SNS

CUDA implementation of the simulation framework from: Hairer, Punshon-Smith, Rosati, Yi (2024)
[arXiv:2411.10419](https://arxiv.org/abs/2411.10419)

Solves the vorticity form of the 2D stochastic Navier-Stokes equations, advects a passive scalar, and computes the top Lyapunov exponent of the tangent equation via Benettin renormalization. Also computes the spectral median and Batchelor scale at each output step.

## Requirements

- CUDA
- ffmpeg

## Build

```bash
chmod +x build.sh
./build.sh
```

## Run

```bash
./solver --Nx 64 --Ny 64 --nu 0.005 --kappa 0.005 --sigma 3.0 \
         --dt 0.0005 --T 30 --si 200 --lr 100 --outdir output
```

## Demo

```bash
python3 visualize.py
```
