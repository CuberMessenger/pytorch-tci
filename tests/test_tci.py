import tqdm
import time
import torch

from pytorch_tci import MultiIndex, BatchedMultiIndex, tensor_cross_interpolation


def compute_relative_error(A, A_tilde):
    return (torch.norm(A - A_tilde) / torch.norm(A)).item()


def compute_absolute_error(A, A_tilde):
    return torch.max(torch.abs(A - A_tilde)).item()


def prepare_random_tensor(size, rank):
    dimension = len(size)

    tensor = torch.randn(rank)
    for i in range(dimension):
        tensor = torch.transpose(tensor, 0, i)
        tensor = torch.tensordot(torch.randn(size[i], rank[i]), tensor, dims=([1], [0]))
        tensor = torch.transpose(tensor, 0, i)

    return tensor


def prepare_asymptotically_smooth_tensor(size, point_dim) -> torch.Tensor:
    pass


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


def main():
    size = (13, 14, 15, 16, 17)
    rank = (3, 4, 5, 6, 7)

    size = (5, 6, 7, 8, 9)
    rank = (2, 2, 7, 4, 4)

    size = (4, 4, 4, 4, 4, 4, 4, 4, 4)
    rank = (2, 2, 2, 2, 2, 2, 2, 2, 2)

    tensor = prepare_random_tensor(size, rank)
    method = "full"

    result = test_tci_single(tensor, method)


if __name__ == "__main__":
    # main()
    size = (5, 6, 7, 8, 9)
    prepare_asymptotically_smooth_tensor(size, point_dim=3)
