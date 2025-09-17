import tqdm
import time
import torch
from pytorch_tci import cross_interpolation


def compute_relative_error(A, A_tilde):
    return (torch.norm(A - A_tilde) / torch.norm(A)).item()


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

    I, J, (_, _, _, query_interpolation_matrix) = cross_interpolation(
        matrix=matrix, method=method, error_threshold=1e-2
    )

    torch.cuda.synchronize()
    end_time = time.perf_counter_ns()

    time_cost = (end_time - start_time) / 1e6  # ms

    # memory
    peak_allocated = torch.cuda.max_memory_allocated(device)

    try:
        pivots_inverse = torch.linalg.inv(matrix[I, :][:, J])
        inv_relative_error = compute_relative_error(
            matrix, matrix[:, J] @ pivots_inverse @ matrix[I, :]
        )
    except Exception as e:
        print(e)
        inv_relative_error = float("inf")

    relative_error = compute_relative_error(matrix, query_interpolation_matrix())

    result = {
        "num_pivots": len(I),
        "I": I,
        "J": J,
        "inv_relative_error": inv_relative_error,
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

    num_pivots = torch.Tensor([len(r["I"]) for r in results])
    inv_relative_errors = torch.Tensor([r["inv_relative_error"] for r in results])
    relative_errors = torch.Tensor([r["relative_error"] for r in results])
    time_costs = torch.Tensor([r["time_cost"] for r in results])
    memory_costs = torch.Tensor([r["memory_cost"] for r in results])

    print(f"Results of testing {method} for {num_iterations} iterations:")
    print(f"Number of pivots:\t{num_pivots.mean().item()} ± {num_pivots.std().item()}")
    print(
        f"Inverse relative error (%):\t{inv_relative_errors.mean().item() * 100:.2f} ± {inv_relative_errors.std().item() * 100:.2f}"
    )
    print(
        f"Relative error (%):\t{relative_errors.mean().item() * 100:.2f} ± {relative_errors.std().item() * 100:.2f}"
    )
    print(
        f"Time cost (ms):\t\t{time_costs.mean().item():.2f} ± {time_costs.std().item():.2f}"
    )
    print(
        f"Memory cost (MB):\t{memory_costs.mean().item():.2f} ± {memory_costs.std().item():.2f}"
    )


def main():
    # N, r = 30, 10
    # test_ci(N, r, method="rook", num_iterations=1)
    # test_ci_single(prepare_test_matrix(N, r).cuda(), method="rook")

    N, r = 240, 60
    # N, r = 1000, 800
    # N, r = 4000, 500
    # N, r = 8000, 400

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
