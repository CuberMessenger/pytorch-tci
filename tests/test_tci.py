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


def prepare_asymptotically_smooth_tensor(
    size: tuple[int], spatial_dim: int
) -> torch.Tensor:
    """
    Prepares a tensor of given size using an asymptotically smooth kernel applied to multiple sets of points in a `spatial_dim`-dimensional space.
    """

    # 0. Parameters
    d = len(size)

    coefficients = [1] * d  # non-zero
    coefficients[-1] = -1
    # coefficients = torch.rand(d) + 0.5  # random coefficients in [0.5, 1.5)

    """
    In 2D, diam(D_Y) <= eta * dist(D_X, D_Y)
    """
    eta = 0.99  # higher eta means looser separation

    """
    C is the offset to the sum of the clusters
    C >= s_pos + s_neg / eta
    """

    # Calculate the domain bounds based on the coefficients
    s_pos = sum(c for c in coefficients if c > 0)
    s_neg = sum(abs(c) for c in coefficients if c < 0)

    C = s_pos + s_neg / eta
    print(f"Current eta ({eta:.2f}) requires C at least to be {C:.2f}")
    C = C * 1.1
    print(f"Setting C to {C:.2f} to satisfy the admissibility condition.")

    # 1. Generate independent clusters of points.
    Xs = []
    for k in range(d):
        Xs.append(torch.rand((size[k], spatial_dim)))

    # 2. Reshape by broadcasting
    for k in range(d):
        view_shape = [1] * (d + 1)
        view_shape[k] = size[k]
        view_shape[-1] = spatial_dim
        Xs[k] = Xs[k].view(*view_shape)

    # 3. Compute the combined vector V for all grid points.
    V = 0
    for k in range(d):
        V = V + coefficients[k] * Xs[k]
    # (n_1, n_2, ..., n_d, spatial_dim)

    # 4. Enforce separation (admissibility condition)
    # offset on the clusters away to guarantee the combined denominator never hits 0.
    V += torch.tensor([C] * spatial_dim).view(*([1] * d), spatial_dim)

    # Compute the Euclidean norm along the spatial dimension (dim=-1)
    dist_tensor = torch.norm(V, p=2.0, dim=-1)
    # (n_1, n_2, ..., n_d)

    # 5. Apply the asymptotically smooth kernel
    kernel_function_selection = 2  # 1 or 2

    if kernel_function_selection == 1:
        # 1 / (4 * pi * r)
        T = 1.0 / (4 * torch.pi * dist_tensor)
    elif kernel_function_selection == 2:
        # cos(k * r) / r
        k = 4.0
        T = torch.cos(k * dist_tensor) / dist_tensor
    else:
        raise ValueError("Invalid kernel function selection. Choose 1 or 2.")

    return T


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


def debug_tci_smooth():
    size = (5, 6, 7, 8, 9)
    size = (4, 4, 4, 4, 4, 4, 4, 4, 4)
    size = tuple([4] * 14) # 4^14 = 268435456; 4^15 = 1073741824 (OOM)
    spatial_dim = 3

    tensor = prepare_asymptotically_smooth_tensor(size, spatial_dim)
    method = "rook"

    result = test_tci_single(tensor, method)


def main():
    debug_tci_smooth()


if __name__ == "__main__":
    main()
