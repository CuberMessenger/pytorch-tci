import torch
from typing import List, Tuple, Callable, Optional, Union

MultiIndex = tuple[Union[int, slice], ...]

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
        query_tensor_element = lambda i: tensor[i]
        query_tensor_superblock = lambda i, j: tensor[i + (slice(None), slice(None)) + j]
        query_tensor = lambda: tensor
        size = tensor.size()

    assert size is not None, "size must be provided."
    dimension = len(size)
    assert dimension >= 3, "tensor must have at least 3 dimensions."

    match method:
        case "full":
            assert query_tensor is not None, "query_tensor must be provided."

            ...

        case "rook":
            assert query_tensor_element is not None, "query_tensor_element must be provided."
            assert query_tensor_superblock is not None, "query_tensor_superblock must be provided."

            ...

        case _:
            raise ValueError(f"Unknown method: {method}")
        
    element = query_tensor_element((0,) * dimension)
    device = element.device
    dtype = element.dtype

    Is = [list() for _ in range(dimension)]
    Js = [list() for _ in range(dimension)]

    while True:
        # left-to-right sweep
        for k in range(dimension - 1):
            # find a new pivot
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

        