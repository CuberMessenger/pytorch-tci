import os
import tqdm
import time
import torch
import matplotlib.pyplot as plot
import matplotlib.ticker as mtick

from pytorch_tci import cross_interpolation

from pytorch_tci.utility import (
    compute_relative_error,
    compute_absolute_error,
    prepare_random_matrix,
    prepare_asymptotically_smooth_tensor,
)


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
        inv_absolute_error = float("inf")

    query_approximation = query_interpolation_matrix()
    relative_error = compute_relative_error(matrix, query_approximation)
    absolute_error = compute_absolute_error(matrix, query_approximation)

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
        prepare_test_matrix = lambda: prepare_asymptotically_smooth_tensor(
            (N, N), dim=5
        )

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


def plot_logs(logs, method, matrix):
    # Parse logs data (skipping header)
    iterations = [row[0] for row in logs[1:]]
    abs_errors = [row[5] for row in logs[1:]]
    crs = [row[6] * 100 for row in logs[1:]]

    fig1, ax1 = plot.subplots(figsize=(8, 6))

    color = "tab:blue"
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Max Absolute Error", color=color)
    ax1.plot(iterations, abs_errors, color=color, label="Absolute Error")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_yscale("log")

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Compression Ratio", color=color)
    ax2.plot(iterations, crs, color=color, linestyle="--", label="CR")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

    ax1.set_title(
        f"Cross Interpolation Error and CR ({method.capitalize()}, N={matrix.size(0)})"
    )
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    fig1.tight_layout()
    plot.savefig(
        os.path.join(
            "outputs", f"ci-{method}-{matrix.size(0)}-{matrix.size(1)}-error-cr.png"
        ),
        dpi=330,
    )
    plot.close(fig1)

    # Singular values subplot
    fig2, ax3 = plot.subplots(figsize=(6, 6))
    S = torch.linalg.svdvals(matrix)
    # Ensure it's sorted descending and on CPU
    S = torch.sort(S, descending=True).values.cpu().numpy()

    color = "tab:green"
    ax3.plot(S, color=color, label="Singular Values")
    ax3.set_yscale("log")
    ax3.set_xlabel("Index")
    ax3.set_ylabel("Singular Value", color=color)
    ax3.tick_params(axis="y", labelcolor=color)
    ax3.set_title("Singular Values of the Original Matrix")
    ax3.grid(True, which="both", ls="--", alpha=0.5)

    fig2.tight_layout()
    # plot.show()
    plot.savefig(
        os.path.join("outputs", f"ci-{method}-{matrix.size(0)}-{matrix.size(1)}.png"),
        dpi=330,
    )
    plot.close(fig2)


def debug_ci(N, r, test_type, method):
    if test_type == "random":
        prepare_test_matrix = lambda: prepare_random_matrix(N, r)
    elif test_type == "smooth":
        prepare_test_matrix = lambda: prepare_asymptotically_smooth_tensor(
            (N, N), spatial_dim=5
        )

    matrix = prepare_test_matrix().cuda()
    # result = test_ci_single(matrix, method)
    # print(result)

    I, J, (_, _, _, query_interpolation_matrix), logs = cross_interpolation(
        matrix=matrix, method=method, error_threshold=1e-16, debug=True
    )

    plot_logs(logs, method, matrix)


def main():
    N, r = 1024, 128
    test_type = "random"
    method = "full"
    debug_ci(N, r, test_type, method)

    # # N, r = 240, 120
    # N, r = 1000, 800
    # # N, r = 4000, 500
    # # N, r = 8000, 4000

    # test_ci(N, r, method="full", num_iterations=10, test_type="smooth")
    # test_ci(N, r, method="rook", num_iterations=10, test_type="smooth")


if __name__ == "__main__":
    main()
