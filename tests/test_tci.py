import tqdm
import time
import torch
import matplotlib.pyplot as plot

from pytorch_tci import MultiIndex, BatchedMultiIndex, tensor_cross_interpolation
from pytorch_tci.utility import (
    compute_relative_error,
    compute_absolute_error,
    prepare_random_tensor,
    prepare_asymptotically_smooth_tensor,
)


def test_tci_single(tensor, method):
    device = torch.device("cuda")
    tensor = tensor.to(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    start_allocated = torch.cuda.memory_allocated(device)

    # timing
    torch.cuda.synchronize()
    start_time = time.perf_counter_ns()

    Is, Js, cores, query_interpolation_element, query_interpolation_tensor = (
        tensor_cross_interpolation(tensor=tensor, method=method, error_threshold=1e-3)
    )

    torch.cuda.synchronize()
    end_time = time.perf_counter_ns()

    time_cost = (end_time - start_time) / 1e6  # ms

    # memory
    peak_allocated = torch.cuda.max_memory_allocated(device)

    tensor_approximation = query_interpolation_tensor()
    relative_error = compute_relative_error(tensor, tensor_approximation)
    absolute_error = compute_absolute_error(tensor, tensor_approximation)

    # statistics
    size = list(tensor.size())
    dimension = len(size)

    rank = [1] + [len(Is[k][0]) for k in range(1, dimension)] + [1]
    original_params = 1
    for s in size:
        original_params *= s

    kept_params = 0
    for core in cores:
        kept_params += core.size(0) * core.size(1) * core.size(2)

    print("=" * 50)
    print("Runtime and Memory Usage " + "-" * 25)
    print(f"          Time cost: {time_cost:.2f} ms")
    print(f"        Memory cost: {(peak_allocated - start_allocated) / 1e6:.2f} MB")
    print("TCI Statistics " + "-" * 35)
    print(f"               Size: {list(size)}")
    print(f"               Rank: {rank}")
    print(f"   Parameter before: {original_params}")
    print(f"   Parameter  after: {kept_params}")
    print(f"  Compression ratio: {kept_params / original_params:.3%}")
    print(f"     Relative error: {relative_error:.3%}")
    print(f"     Absolute error: {absolute_error:.3e}")
    print("=" * 50)

    result = {
        "rank": rank,
        "Is": Is,
        "Js": Js,
        "cores": cores,
        "relative_error": relative_error,
        "absolute_error": absolute_error,
        "time_cost": time_cost,
        "memory_cost": (peak_allocated - start_allocated) / 1e6,
    }

    return result


def debug_tci_random():
    size = (13, 14, 15, 16, 17)
    rank = (3, 4, 5, 6, 7)

    size = (5, 6, 7, 8, 9)
    rank = (2, 2, 7, 4, 4)

    size = (4, 4, 4, 4, 4, 4, 4, 4, 4)
    rank = (2, 2, 2, 2, 2, 2, 2, 2, 2)

    tensor = prepare_random_tensor(size, rank)
    method = "rook"

    result = test_tci_single(tensor, method)


def plot_logs(logs, method, tensor):
    # Parse logs data (skipping header)
    iterations = [row[0] for row in logs[1:]]
    abs_errors = [row[-2] for row in logs[1:]]
    crs = [row[-1] for row in logs[1:]]

    fig, (ax1, ax3) = plot.subplots(1, 2, figsize=(12, 6))

    color = "tab:blue"
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Absolute Error", color=color)
    ax1.plot(iterations, abs_errors, color=color, label="Absolute Error")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_yscale("log")

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Compression Ratio (%)", color=color)
    ax2.plot(iterations, crs, color=color, linestyle="--", label="CR")
    ax2.tick_params(axis="y", labelcolor=color)
    ax1.set_title(
        f"Tensor Cross Interpolation Error and CR ({method.capitalize()}, {list(tensor.size())})"
    )
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    i = 100
    superblock = logs[i][-4]
    r_l, d_l, d_r, r_r = tuple(superblock.size())
    superblock = superblock.reshape(r_l * d_l, d_r * r_r)

    # Singular values subplot
    S = torch.linalg.svdvals(superblock)
    # Ensure it's sorted descending and on CPU
    S = torch.sort(S, descending=True).values.cpu().numpy()

    color = "tab:green"
    ax3.plot(S, color=color, label="Singular Values")
    ax3.set_yscale("log")
    ax3.set_xlabel("Index")
    ax3.set_ylabel("Singular Value", color=color)
    ax3.tick_params(axis="y", labelcolor=color)
    ax3.set_title(
        f"Singular Values of the {i}-th Superblock ({method.capitalize()}, {list(superblock.size())})"
    )
    ax3.grid(True, which="both", ls="--", alpha=0.5)

    fig.tight_layout()
    plot.show()


def debug_tci_smooth():
    size = (5, 6, 7, 8, 9)
    size = (4, 4, 4, 4, 4, 4, 4, 4, 4)
    size = tuple([4] * 14)  # 4^14 = 268435456; 4^15 = 1073741824 (OOM)
    size = (14, 15, 16, 17, 18)
    spatial_dim = 5

    tensor = prepare_asymptotically_smooth_tensor(size, spatial_dim)
    tensor = tensor.cuda()
    method = "rook"

    Is, Js, cores, query_interpolation_element, query_interpolation_tensor, logs = (
        tensor_cross_interpolation(
            tensor=tensor, method=method, error_threshold=1e-6, debug=True
        )
    )

    plot_logs(logs, method, tensor)


def main():
    debug_tci_smooth()


if __name__ == "__main__":
    main()
