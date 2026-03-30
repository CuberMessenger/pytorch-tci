import os
import tqdm
import time
import torch
import matplotlib.pyplot as plot
import matplotlib.ticker as mtick


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
    crs = [row[-1] * 100 for row in logs[1:]]

    tensor_size_str = "-".join([str(s) for s in tensor.size()])

    # --- 1) Plot and save Error and CR ---
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
        f"Tensor Cross Interpolation Error and CR ({method.capitalize()}, {list(tensor.size())})"
    )
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    fig1.tight_layout()
    plot.savefig(
        os.path.join("outputs", f"tci-{method}-{tensor_size_str}-error-cr.png"), dpi=330
    )
    plot.close(fig1)

    # --- 2) Plot and save singular values for each superblock ---
    for i in range(1, 11):
        if i * 100 >= len(logs):
            break

        fig2, ax3 = plot.subplots(figsize=(6, 6))

        superblock = logs[i * 100][-4]
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

        fig2.tight_layout()
        plot.savefig(
            os.path.join("outputs", f"tci-{method}-it{i}-superblock-{r_l * d_l}-{d_r * r_r}.png"), dpi=330
        )
        plot.close(fig2)


def debug_tci(test_type):
    if test_type == "random":
        size = (16, 16, 16, 16, 16)
        rank = (5, 6, 7, 8, 9)
        tensor = prepare_random_tensor(size, rank)
    else:
        size = (5, 6, 7, 8, 9)
        size = (4, 4, 4, 4, 4, 4, 4, 4, 4)
        size = tuple([4] * 14)  # 4^14 = 268435456; 4^15 = 1073741824 (OOM)
        size = (16, 16, 16, 16, 16)
        spatial_dim = 5

        tensor = prepare_asymptotically_smooth_tensor(size, spatial_dim)
    tensor = tensor.cuda()
    method = "full"

    # torch.cuda.synchronize()
    # start_time = time.perf_counter_ns()

    # Is, Js, cores, query_interpolation_element, query_interpolation_tensor = (
    #     tensor_cross_interpolation(
    #         tensor=tensor, method=method, error_threshold=1e-6, debug=False
    #     )
    # )
    # torch.cuda.synchronize()
    # end_time = time.perf_counter_ns()
    # time_cost = (end_time - start_time) / 1e6  # ms
    # print(f"TCI completed in {time_cost:.2f} ms")
    # # TCI completed in 12899.78 ms for size (16, 16, 16, 16, 16) and spatial_dim=5

    Is, Js, cores, query_interpolation_element, query_interpolation_tensor, logs = (
        tensor_cross_interpolation(
            tensor=tensor, method=method, error_threshold=1e-16, debug=True
        )
    )

    plot_logs(logs, method, tensor)


def main():
    debug_tci("random")


if __name__ == "__main__":
    main()
