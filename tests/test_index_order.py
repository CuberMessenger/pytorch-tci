import torch, time


def bench(m=20000, n=1024, k=5000, device="cuda"):
    A = torch.randn(m, n, device=device)
    I_rand = torch.randperm(m, device=device)[:k]
    I_sorted = torch.sort(I_rand).values

    # warmup
    for _ in range(5):
        _ = A[I_rand, :] * 2
        _ = A[I_sorted, :] * 2
    if device == "cuda":
        torch.cuda.synchronize()

    def timeit(I, reps=20):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            _ = A[I, :] * 2
        if device == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps

    t_rand = timeit(I_rand)
    t_sorted = timeit(I_sorted)
    print(f"rand:   {t_rand*1e3:.2f} ms")
    print(f"sorted: {t_sorted*1e3:.2f} ms")
    print(f"speedup: {(t_rand/t_sorted):.2f}×")


if __name__ == "__main__":
    bench()
