import torch

from typing import List, Tuple, Callable, Optional, Union

MultiIndex = tuple[Union[int, slice], ...]

def eqrange(a: int, b: int) -> range:
    return range(a, b + 1)

def tensor_cross_interpolation(
    query_tensor_element: Optional[Callable[[MultiIndex], torch.Tensor]] = None,
    query_tensor_superblock: Optional[Callable[[MultiIndex, MultiIndex], torch.Tensor]] = None,
    query_tensor: Optional[Callable[[], torch.Tensor]] = None,
    size: Optional[list[int]] = None,
    tensor: Optional[torch.Tensor] = None,
    method: str = "rook",
    error_threshold: float = 1e-3,
):
    if tensor is not None:
        query_tensor_element = lambda I_a: tensor[I_a]
        query_tensor_superblock = lambda I_km1, J_kp2: tensor[I_km1 + (slice(None), slice(None)) + J_kp2]
        query_tensor = lambda: tensor
        size = tensor.size()

    assert size is not None, "size must be provided."
    dimension = len(size)
    assert dimension >= 3, "tensor must have at least 3 dimensions."
    assert query_tensor_element is not None, "query_tensor_element must be provided."
    assert query_tensor_superblock is not None, "query_tensor_superblock must be provided."

    match method:
        case "full":
            ...

        case "rook":
            ...

        case _:
            raise ValueError(f"Unknown method: {method}")
        
    element = query_tensor_element((0,) * dimension)
    device = element.device
    dtype = element.dtype

    Is = [()] + [(0,) * k for k in eqrange(1, dimension - 1)]
    #    I_0     I_1, ..., I_{d-1}
    #   Is[0]   Is[1]      Is[d-1]
    Js = [None, None] + [(0,) * (dimension - k) for k in eqrange(2, dimension)] + [()]
    #                    J_2, ..., J_d                                            J_{d+1}
    #                   Js[2]      Js[d]                                          Js[d+1]
    while True:
        # left-to-right sweep
        for k in eqrange(1, dimension - 1):
            # find a new pivot
            ## form the supercore
            supercore = query_tensor_superblock(Is[k - 1], Js[k + 2]) # (r_{k - 1} n_k, n_{k + 1} r_{k + 2})

            ## form the supercore approximation using current Is, Js
            supercore_approx = ...

            ## apply cross interpolation to the supercore
            ...

            # update Is[k], Js[k]
            ...

        # right-to-left sweep
        for k in reversed(range(1, dimension)):
            # find a new pivot
            ...

            # update Is[k], Js[k]
            ...

        # check convergence
        ...

        