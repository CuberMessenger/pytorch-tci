import tqdm
import time
import torch
from pytorch_tci import ci


def prepare_test_matrix(N, r):
    return torch.rand((N, r)) @ torch.rand((r, N))  # + torch.rand((N, N)) * 1e-2


def test_ci_single(matrix, method):
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    start_allocated = torch.cuda.memory_allocated(device)

    # timing
    torch.cuda.synchronize()
    start_time = time.perf_counter_ns()

    I, J, _ = ci(matrix, method=method, error_threshold=1e-2)

    torch.cuda.synchronize()
    end_time = time.perf_counter_ns()

    time_cost = (end_time - start_time) / 1e6  # ms

    # memory
    peak_allocated = torch.cuda.max_memory_allocated(device)

    try:
        pivots_inverse = torch.linalg.inv(matrix[I, :][:, J])
    except Exception as e:
        print(e)
        return {}

    relative_error = (
        torch.norm(matrix - matrix[:, J] @ pivots_inverse @ matrix[I, :])
        / torch.norm(matrix)
    ).item()

    result = {
        "num_pivots": len(I),
        "I": I,
        "J": J,
        "relative_error": relative_error,
        "time_cost": time_cost,
        "memory_cost": (peak_allocated - start_allocated) / 1e6,
    }

    return result


def test_ci(N, r, method, num_iterations):
    results = []

    # warm up
    for _ in tqdm.tqdm(range(3), desc=f"Warm up {method} ......"):
        matrix = prepare_test_matrix(N, r).cuda()
        _ = test_ci_single(matrix, method)

    for _ in tqdm.tqdm(range(num_iterations), desc=f"Testing {method} ......"):
        matrix = prepare_test_matrix(N, r).cuda()
        result = test_ci_single(matrix, method)
        results.append(result)

    relative_errors = torch.Tensor([r["relative_error"] for r in results])
    time_costs = torch.Tensor([r["time_cost"] for r in results])
    memory_costs = torch.Tensor([r["memory_cost"] for r in results])

    print(f"Results of testing {method} for {num_iterations} iterations:")
    print(
        f"Relative error (%):\t{relative_errors.mean().item() * 100} ± {relative_errors.std().item() * 100}"
    )
    print(f"Time cost (ms):\t\t{time_costs.mean().item()} ± {time_costs.std().item()}")
    print(
        f"Memory cost (MB):\t{memory_costs.mean().item()} ± {memory_costs.std().item()}"
    )


def main():
    # N, r = 30, 10
    # test_ci(N, r, method="rook", num_iterations=1)
    # test_ci_single(prepare_test_matrix(N, r).cuda(), method="rook")

    # N, r = 240, 60
    N, r = 15000, 200

    test_ci(N, r, method="full", num_iterations=4)
    test_ci(N, r, method="rook", num_iterations=4)

    """
    N, r = 240, 60

    []
    Results of testing full for 16 iterations:
    Relative error (%):     0.0016677120584063232 ± 0.0005922380751144374
    Time cost (ms):         48.1142578125 ± 2.977712631225586
    Memory cost (MB):       0.8099840879440308 ± 6.155941179031288e-08
    Results of testing rook for 16 iterations:
    Relative error (%):     0.2969968132674694 ± 1.180540770292282
    Time cost (ms):         123.07276153564453 ± 19.904434204101562
    Memory cost (MB):       0.22867199778556824 ± 0.032856881618499756
    """


if __name__ == "__main__":
    main()
