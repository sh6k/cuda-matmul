# CUDA Matrix Multiplication

Parallel matrix multiplication implemented in CUDA using PyCUDA, benchmarked against NumPy CPU on an NVIDIA Tesla T4 GPU.

## Results

| Matrix Size | GPU Time | CPU Time | Speedup |
|-------------|----------|----------|---------|
| 512x512     | 1.25ms   | 6.03ms   | 4.8x    |
| 1024x1024   | 9.31ms   | 44.09ms  | 4.7x    |
| 2048x2048   | 75.01ms  | 287.70ms | 3.8x    |

## How it works

Each element of the result matrix C is computed by an independent CUDA thread.
Threads are organized into 16x16 blocks across a 2D grid covering the full matrix.
Data is explicitly transferred between CPU and GPU memory before and after computation.

## Run it

Open in Google Colab with a T4 GPU runtime:

1. Runtime → Change runtime type → T4 GPU
2. Run all cells
