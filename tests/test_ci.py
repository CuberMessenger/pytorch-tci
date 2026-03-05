import tqdm
import time
import torch
from pytorch_tci import cross_interpolation


def compute_relative_error(A, A_tilde):
    return (torch.norm(A - A_tilde) / torch.norm(A)).item()


def prepare_random_matrix(N, r):
    return torch.rand((N, r)) @ torch.rand((r, N))  # + torch.rand((N, N)) * 1e-2


def prepare_asymptotically_smooth_matrix(num_points, dim) -> torch.Tensor:
    """
    Prepares two well-separated clusters of points in 3D space to satisfy
    the geometrical condition for asymptotically smooth functions.
    """
    # Cluster X centered around the origin point
    X = torch.rand((num_points, dim))

    # Cluster Y centered around a distant point to ensure separation
    # This separation ensures |x - y| is never 0, avoiding singularities,
    # and satisfies the eta-admissibility condition.
    Y = torch.rand((num_points, dim)) + torch.tensor([3] * dim)

    """
    Prepares the matrix A based on the smooth operator kernel:
    f1(x, y) = 1 / (4 * pi * |x - y|)
    f2(x, y) = cos(k * |x - y|) / |x - y|
    """
    kernel_function_selection = 2  # 1 or 2

    # Compute pairwise Euclidean distances between points in X and Y
    # dist_matrix shape: [num_points_x, num_points_y]
    dist_matrix = torch.cdist(X, Y, p=2.0)

    # The asymptotically smooth function
    if kernel_function_selection == 1:
        A = 1.0 / (4 * torch.pi * dist_matrix)
    elif kernel_function_selection == 2:
        k = 2.0
        A = torch.cos(k * dist_matrix) / dist_matrix
    else:
        raise ValueError("Invalid kernel function selection. Choose 1 or 2.")

    return A


def test_ci_single(matrix, method):
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    start_allocated = torch.cuda.memory_allocated(device)

    # timing
    torch.cuda.synchronize()
    start_time = time.perf_counter_ns()

    I, J, (_, _, _, query_interpolation_matrix) = cross_interpolation(
        matrix=matrix, method=method, error_threshold=1e-6
    )

    torch.cuda.synchronize()
    end_time = time.perf_counter_ns()

    time_cost = (end_time - start_time) / 1e6  # ms

    # memory
    peak_allocated = torch.cuda.max_memory_allocated(device)

    try:
        pivots_inverse = torch.linalg.inv(matrix[I, :][:, J])
        inv_approximation = matrix[:, J] @ pivots_inverse @ matrix[I, :]
        inv_absolute_error = torch.max(torch.abs(matrix - inv_approximation))
        inv_relative_error = compute_relative_error(matrix, inv_approximation)
    except Exception as e:
        print(e)
        inv_relative_error = float("inf")

    query_approximation = query_interpolation_matrix()
    relative_error = compute_relative_error(matrix, query_approximation)
    absolute_error = torch.max(torch.abs(matrix - query_approximation))

    result = {
        "num_pivots": len(I),
        "I": I,
        "J": J,
        "inv_relative_error": inv_relative_error,
        "inv_absolute_error": inv_absolute_error,
        "relative_error": relative_error,
        "absolute_error": absolute_error,
        "time_cost": time_cost,
        "memory_cost": (peak_allocated - start_allocated) / 1e6,
    }

    return result


def test_ci(N, r, method, num_iterations, test_type="random"):
    """
    test_type: "random" or "smooth"
    """
    results = []

    if test_type == "random":
        prepare_test_matrix = lambda: prepare_random_matrix(N, r)
    elif test_type == "smooth":
        prepare_test_matrix = lambda: prepare_asymptotically_smooth_matrix(N, dim=5)

    # warm up
    for _ in tqdm.tqdm(range(3), desc=f"Warm up {method} ......"):
        matrix = prepare_test_matrix().cuda()
        _ = test_ci_single(matrix, method)

    for _ in tqdm.tqdm(range(num_iterations), desc=f"Testing {method} ......"):
        matrix = prepare_test_matrix().cuda()
        result = test_ci_single(matrix, method)
        results.append(result)

    num_pivots = torch.Tensor([len(r["I"]) for r in results])
    inv_relative_errors = torch.Tensor([r["inv_relative_error"] for r in results])
    inv_absolute_errors = torch.Tensor([r["inv_absolute_error"] for r in results])
    relative_errors = torch.Tensor([r["relative_error"] for r in results])
    absolute_errors = torch.Tensor([r["absolute_error"] for r in results])
    time_costs = torch.Tensor([r["time_cost"] for r in results])
    memory_costs = torch.Tensor([r["memory_cost"] for r in results])

    print(f"")
    print(f"Results of testing {method} for {num_iterations} iterations:")
    print(
        f"Number of pivots:\t{num_pivots.mean().item():.2f} ± {num_pivots.std().item():.2f}"
    )
    print(
        f"Inv. relative error:\t{inv_relative_errors.mean().item() * 100:.2f} ± {inv_relative_errors.std().item() * 100:.2f} %"
    )
    print(
        f"Inv. absolute error:\t{inv_absolute_errors.mean().item():.3e} ± {inv_absolute_errors.std().item():.3e}"
    )
    print(
        f"Relative error:\t\t{relative_errors.mean().item() * 100:.2f} ± {relative_errors.std().item() * 100:.2f} %"
    )
    print(
        f"Absolute error:\t\t{absolute_errors.mean().item():.3e} ± {absolute_errors.std().item():.3e}"
    )
    print(
        f"Time cost:\t\t{time_costs.mean().item():.2f} ± {time_costs.std().item():.2f} ms"
    )
    print(
        f"Memory cost:\t\t{memory_costs.mean().item():.2f} ± {memory_costs.std().item():.2f} MB"
    )


def main():
    # N, r = 30, 10
    # test_ci(N, r, method="rook", num_iterations=1)
    # test_ci_single(prepare_test_matrix(N, r).cuda(), method="rook")

    # N, r = 240, 120
    N, r = 1000, 800
    # N, r = 4000, 500
    # N, r = 8000, 4000

    test_ci(N, r, method="full", num_iterations=10, test_type="smooth")
    test_ci(N, r, method="rook", num_iterations=10, test_type="smooth")

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
