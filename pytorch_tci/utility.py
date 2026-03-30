import torch


def compute_relative_error(A, A_tilde):
    return (torch.norm(A - A_tilde) / torch.norm(A)).item()


def compute_absolute_error(A, A_tilde):
    return torch.max(torch.abs(A - A_tilde)).item()


def prepare_random_matrix(N, r):
    return torch.rand((N, r)) @ torch.rand((r, N))  # + torch.randn((N, N)) * 1e-3


def prepare_random_tensor(size, rank):
    dimension = len(size)

    tensor = torch.randn(rank)
    for i in range(dimension):
        tensor = torch.transpose(tensor, 0, i)
        tensor = torch.tensordot(torch.randn(size[i], rank[i]), tensor, dims=([1], [0]))
        tensor = torch.transpose(tensor, 0, i)
    # tensor += torch.randn(size) * 1e-3  # add small noise

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
    eta = 0.9  # higher eta means looser separation

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
