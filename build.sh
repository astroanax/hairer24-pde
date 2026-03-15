#!/bin/bash
export PATH="/opt/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/opt/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
nvcc -O3 -std=c++17 -arch=sm_89 -o solver solver.cu -lcufft -lcurand -lm
echo "Built: $(ls -la solver)"
