sizes = [512, 1024, 2048]

for N in sizes:
    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)
    C_gpu = np.zeros((N, N), dtype=np.float32)

    A_gpu = cuda.mem_alloc(A.nbytes)
    B_gpu = cuda.mem_alloc(B.nbytes)
    C_gpu_mem = cuda.mem_alloc(C_gpu.nbytes)

    cuda.memcpy_htod(A_gpu, A)
    cuda.memcpy_htod(B_gpu, B)

    block_size = 16
    grid_size = (N + block_size - 1) // block_size

    start = time.time()
    matmul_gpu(
        A_gpu, B_gpu, C_gpu_mem,
        np.int32(N),
        block=(block_size, block_size, 1),
        grid=(grid_size, grid_size)
    )
    cuda.Context.synchronize()
    gpu_time = time.time() - start

    cuda.memcpy_dtoh(C_gpu, C_gpu_mem)

    start = time.time()
    C_cpu = np.dot(A, B)
    cpu_time = time.time() - start

    match = np.allclose(C_gpu, C_cpu, atol=1e-3)
    print(f"N={N:4d} | GPU: {gpu_time*1000:7.2f}ms | CPU: {cpu_time*1000:7.2f}ms | Speedup: {cpu_time/gpu_time:5.1f}x | Match: {match}")
