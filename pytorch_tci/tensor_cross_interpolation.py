import os
import torch

from typing import Callable, Optional
from enum import Enum, auto
from itertools import chain

MultiIndex = tuple[int | slice, ...]
BatchedMultiIndex = tuple[list[list[int]] | slice, ...]
"""
For left-side batched multi-index, each dimension either:
    a slice, or
    a list of lists of integers, where the inner list always contains exact one int.

For right-side batched multi-index, each dimension either:
    a slice, or
    a list of lists of integers, where the outer list always contains exact one list.

For example,
left = ([[2], [2]], [[0], [1]]) # two points (2, 0) and (2, 1) taking care of the first two dimensions
right = ([[2, 3, 3]], [[1, 4, 5]]) # three points (2, 1), (3, 4), and (3, 5) taking care of the last two dimensions
"""


def eqrange(a: int, b: int) -> range:
    return range(a, b + 1)


def query_tensor_superfiber(
    Is: MultiIndex,
    Js: MultiIndex,
    query_tensor_element: Callable[[MultiIndex], torch.Tensor] = None,
    query_tensor_element_batched: Callable[[BatchedMultiIndex], torch.Tensor] = None,
):
    if query_tensor_element_batched is None:
        raise NotImplementedError("Non-batched mode is not implemented yet.")

    if len(Is) == 0:
        superfiber = query_tensor_element_batched((slice(None),) + Js)
        # (free_dim, 1, batch_J)
        superfiber = superfiber.permute((1, 0, 2))
        # (1, free_dim, batch_J)
    elif len(Js) == 0:
        superfiber = query_tensor_element_batched(Is + (slice(None),))
        # (batch_I, 1, free_dim)
        superfiber = superfiber.permute((0, 2, 1))
        # (batch_I, free_dim, 1)
    else:
        superfiber = query_tensor_element_batched(Is + (slice(None),) + Js)
        # (batch_I, batch_J, free_dim)
        superfiber = superfiber.permute((0, 2, 1))
        # (batch_I, free_dim, batch_J)

    return superfiber


def query_tensor_superblock(
    Is: MultiIndex,
    Js: MultiIndex,
    query_tensor_element: Callable[[MultiIndex], torch.Tensor] = None,
    query_tensor_element_batched: Callable[[BatchedMultiIndex], torch.Tensor] = None,
):
    if query_tensor_element_batched is None:
        raise NotImplementedError("Non-batched mode is not implemented yet.")

    """
    Format advanced indexing results to ensure the output shape is always
    (batch_I, free_dim1, free_dim2, batch_J).
    
    PyTorch advanced indexing behavior:
    - If batched indices (Is, Js) are separated by slices, they are all moved to the front.
    - If not separated by slices (e.g., Is or Js is empty), the batched indices 
      stay at the position of the first advanced index.
    """

    if len(Is) == 0:
        superblock = query_tensor_element_batched((slice(None), slice(None)) + Js)
        # (free_dim1, free_dim2, 1, batch_J)
        superblock = superblock.permute((2, 0, 1, 3))
        # (1, free_dim1, free_dim2, batch_J)
    elif len(Js) == 0:
        superblock = query_tensor_element_batched(Is + (slice(None), slice(None)))
        # (batch_I, 1, free_dim1, free_dim2)
        superblock = superblock.permute((0, 2, 3, 1))
        # (batch_I, free_dim1, free_dim2, 1)
    else:
        superblock = query_tensor_element_batched(Is + (slice(None), slice(None)) + Js)
        # (batch_I, batch_J, free_dim1, free_dim2)
        superblock = superblock.permute((0, 2, 3, 1))
        # (batch_I, free_dim1, free_dim2, batch_J)

    return superblock


class RookCondition(Enum):
    RULES_ROW = auto()
    RULES_COLUMN = auto()


def full_search(
    error_tensor: torch.Tensor,
) -> tuple[int, int, int, int, torch.Tensor]:
    a_star, i_star, j_star, b_star = torch.unravel_index(
        torch.argmax(error_tensor.abs()), error_tensor.size()
    )
    a_star, i_star, j_star, b_star = (
        a_star.item(),
        i_star.item(),
        j_star.item(),
        b_star.item(),
    )
    ep = error_tensor[a_star, i_star, j_star, b_star]

    return a_star, i_star, j_star, b_star, ep


def rook_search(
    error_tensor: torch.Tensor,
    max_iteration: int = 4,
) -> tuple[int, int, int, int, torch.Tensor]:
    a_star = torch.randint(0, error_tensor.size(0), (1,)).item()
    i_star = torch.randint(0, error_tensor.size(1), (1,)).item()
    j_star = 0
    b_star = 0
    ep = None

    def rook_row():
        nonlocal a_star, i_star, j_star, b_star, ep
        slice_row = error_tensor[a_star, i_star, :, :]
        next_j, next_b = torch.unravel_index(
            torch.argmax(slice_row.abs()), slice_row.size()
        )
        next_j, next_b = next_j.item(), next_b.item()

        moved = (next_j != j_star) or (next_b != b_star)
        j_star, b_star = next_j, next_b
        ep = error_tensor[a_star, i_star, j_star, b_star]

        return moved

    def rook_column():
        nonlocal a_star, i_star, j_star, b_star, ep
        slice_col = error_tensor[:, :, j_star, b_star]
        next_a, next_i = torch.unravel_index(
            torch.argmax(slice_col.abs()), slice_col.size()
        )
        next_a, next_i = next_a.item(), next_i.item()

        moved = (next_a != a_star) or (next_i != i_star)
        a_star, i_star = next_a, next_i
        ep = error_tensor[a_star, i_star, j_star, b_star]

        return moved

    rook_row()
    rook_condition = RookCondition.RULES_ROW

    for _ in range(max_iteration - 1):
        if rook_condition is RookCondition.RULES_ROW:
            moved = rook_column()

            if moved:
                rook_condition = RookCondition.RULES_COLUMN
            else:
                break

            continue

        if rook_condition is RookCondition.RULES_COLUMN:
            moved = rook_row()

            if moved:
                rook_condition = RookCondition.RULES_ROW
            else:
                break

            continue

    ep = error_tensor[a_star, i_star, j_star, b_star]

    return a_star, i_star, j_star, b_star, ep


def sweep_one(
    k,
    Is,
    Js,
    query_tensor_element,
    query_tensor_element_batched,
    searcher,
    error_threshold,
):
    # find a new pivot
    ## form the supercore
    changed = False
    supercore = query_tensor_superblock(
        Is=Is[k - 1],
        Js=Js[k + 2],
        query_tensor_element=query_tensor_element,
        query_tensor_element_batched=query_tensor_element_batched,
    )
    print(f"k = {k}, supercore: {list(supercore.size())}")
    # r_{k - 1} * n_k, n_{k + 1} * r_{k + 2}

    ## form the supercore approximation using current Is, Js

    supercore_approximation_left = query_tensor_superfiber(
        Is[k - 1],
        Js[k + 1],
        query_tensor_element=query_tensor_element,
        query_tensor_element_batched=query_tensor_element_batched,
    )
    # r_{k - 1} * n_k * r_{k + 1}

    supercore_approximation_middle = query_tensor_element_batched(Is[k] + Js[k + 1])
    # r_{k} * r_{k + 1}

    supercore_approximation_right = query_tensor_superfiber(
        Is[k],
        Js[k + 2],
        query_tensor_element=query_tensor_element,
        query_tensor_element_batched=query_tensor_element_batched,
    )
    # r_{k} * n_{k + 1} * r_{k + 2}

    supercore_approximation = torch.einsum(
        "apc,cb,bqd->apqd",
        supercore_approximation_left,
        torch.linalg.pinv(supercore_approximation_middle),
        supercore_approximation_right,
    )

    # apply cross interpolation to the supercore
    error_tensor = supercore - supercore_approximation
    a_star, i_star, j_star, b_star, ep = searcher((error_tensor,))

    # update Is[k], Js[k+1]
    if ep.abs() > error_threshold:
        changed = True

        for m in range(k - 1):
            Is[k][m].append([Is[k - 1][m][a_star][0]])
        Is[k][k - 1].append([i_star])

        Js[k + 1][0][0].append(j_star)
        for m in range(dimension - k - 1):
            Js[k + 1][m + 1][0].append(Js[k + 2][m][0][b_star])

    return changed


def tensor_cross_interpolation(
    query_tensor_element: Optional[Callable[[MultiIndex], torch.Tensor]] = None,
    query_tensor_element_batched: Optional[
        Callable[[BatchedMultiIndex], torch.Tensor]
    ] = None,
    query_tensor: Optional[Callable[[], torch.Tensor]] = None,
    size: Optional[list[int]] = None,
    tensor: Optional[torch.Tensor] = None,
    method: str = "rook",
    error_threshold: float = 1e-3,
):
    if tensor is not None:
        query_tensor_element = lambda I_a: tensor[I_a]
        query_tensor_element_batched = lambda I_a: tensor[I_a]
        query_tensor = lambda: tensor
        size = tensor.size()

    assert size is not None, "size must be provided."
    dimension = len(size)
    assert dimension >= 3, "tensor must have at least 3 dimensions."
    assert query_tensor_element is not None, "query_tensor_element must be provided."
    assert (
        query_tensor_superblock is not None
    ), "query_tensor_superblock must be provided."

    match method:
        case "full":
            searcher = lambda args: full_search(*args)

        case "rook":
            searcher = lambda args: rook_search(*args)

        case _:
            raise ValueError(f"Unknown method: {method}")

    element = query_tensor_element((0,) * dimension)
    device = element.device
    dtype = element.dtype

    ## A MultiIndex is a tuple, whose elements are either lists of integers or slices.
    ## Is[k] is one MultiIndex with length k
    ## Js[k] is one MultiIndex with length dimension - k + 1

    #  I_0     I_1  ...  I_{d-1}
    # Is[0]   Is[1] ...  Is[d-1]
    Is = [None] * dimension
    Is[0] = ()
    for k in eqrange(1, dimension - 1):
        Is[k] = tuple([[0]] for _ in range(k))
    Is = tuple(Is)

    #  J_2   ...  J_d    J_{d+1}
    # Js[2]      Js[d]   Js[d+1]
    Js = [None] * (dimension + 2)
    for k in eqrange(2, dimension):
        Js[k] = tuple([[0]] for _ in range(dimension - k + 1))
    Js[-1] = ()
    Js = tuple(Js)

    # """
    # fmt: off
    ### EXAMPLE ###

    # Is[0] = ()
    # Is[1] = ([[0]],)
    # Is[2] = ([[0]], [[0]],)
    # Is[3] = ([[0]], [[0]], [[0]],)
    # Is[4] = ([[0]], [[0]], [[0]], [[0]],)

    # Js[0] = None
    # Js[1] = None
    # Js[2] = ([[0]], [[0]], [[0]], [[0]],)
    # Js[3] = ([[0]], [[0]], [[0]],)
    # Js[4] = ([[0]], [[0]],)
    # Js[5] = ([[0]],)
    # Js[6] = ()

    # ----

    # Is[0] = ()
    # Is[1] = ([[2], [0]],)                                       # (2, ...) and (0, ...)
    # Is[2] = ([[2], [2]], [[0], [1]])                            # (2, 0, ...) and (2, 1, ...)
    # Is[3] = ([[2], [2]], [[0], [1]], [[3], [4]])                # (2, 0, 3, ...) and (2, 1, 4, ...)
    # Is[4] = ([[2], [2]], [[0], [1]], [[3], [4]], [[0], [5]])    # (2, 0, 3, 0, ...) and (2, 1, 4, 5, ...)

    # Js[0] = None
    # Js[1] = None
    # Js[2] = ([[1, 2]], [[2, 0]], [[3, 3]], [[1, 1]])            # (..., 1, 2, 3, 1) and (..., 2, 0, 3, 1)
    # Js[3] = ([[2, 0]], [[3, 3]], [[1, 1]])                      # (..., 2, 3, 1) and (..., 0, 3, 1)
    # Js[4] = ([[3, 4]], [[1, 1]])                                # (..., 3, 1) and (..., 4, 1)
    # Js[5] = ([[1, 4, 5]],)                                      # (..., 1) and (..., 4) and (..., 5)
    # Js[6] = ()

    ### EXAMPLE ###
    # fmt: on
    # """

    changed = True
    while changed:
        changed = False

        # left-to-right sweep then right-to-left sweep
        for k in chain(eqrange(1, dimension - 1), reversed(range(1, dimension - 1))):
            changed |= sweep_one(
                k,
                Is,
                Js,
                query_tensor_element,
                query_tensor_element_batched,
                searcher,
                error_threshold,
            )

    cores = []

    for k in eqrange(1, dimension - 1):
        core_left = query_tensor_superfiber(
            Is[k - 1],
            Js[k + 1],
            query_tensor_element=query_tensor_element,
            query_tensor_element_batched=query_tensor_element_batched,
        )  # (r_{k-1}, n_k, r_{k+1})

        core_right = torch.linalg.pinv(
            query_tensor_element_batched(Is[k] + Js[k + 1])
        )  # (r_k, r_{k+1})

        core = torch.einsum(
            "apc,cb->apb",
            core_left,
            core_right,
        )  # (r_{k-1}, n_k, r_{k+1}) * (r_{k+1}, r_k) -> (r_{k-1}, n_k, r_k)

        print(f"core {k}: {list(core.size())}")

        cores.append(core)

    last_core = query_tensor_superfiber(
        Is[dimension - 1],
        Js[dimension + 1],
        query_tensor_element=query_tensor_element,
        query_tensor_element_batched=query_tensor_element_batched,
    )  # (r_{d-2}, n_{d-1}, 1)

    print(f"core {dimension}: {list(last_core.size())}")
    cores.append(last_core)

    def query_interpolation_element(I: MultiIndex) -> torch.Tensor:
        # element = torch.linalg.multi_dot(
        #     [core[:, index, :] for core, index in zip(cores, I)]
        # ).squeeze()
        # return element

        sliced_cores = [core[:, idx, :] for core, idx in zip(cores, I)]
        result = sliced_cores[0]
        for next_core in sliced_cores[1:]:
            # result: (..., r_k)
            #               [-1]
            # next_core: (r_k, ...)
            #             [0]
            result = torch.tensordot(result, next_core, dims=([-1], [0]))

        return result.squeeze()
    
    def query_interpolation_tensor() -> torch.Tensor:
        mi = tuple(slice(None) for _ in range(dimension))
        return query_interpolation_element(mi)

    mi = (1, 2, 3, 4, 5)
    print(f"query_interpolation_element({mi}): {query_interpolation_element(mi)}")
    print(f"query_tensor_element({mi}): {query_tensor_element(mi)}")

    mi = (slice(None), 2, 3, slice(None), 5)
    interpolated = query_interpolation_element(mi)
    original = query_tensor_element(mi)
    print(f"query_interpolation_element({mi}): {interpolated.size()}")
    print(f"query_tensor_element({mi}): {original.size()}")
    abs_error = (interpolated - original).abs().max()
    print(f"max absolute error: {abs_error.item()}")

    interpolated_tensor = query_interpolation_tensor()
    original_tensor = query_tensor()
    print(f"query_interpolation_tensor(): {interpolated_tensor.size()}")
    print(f"query_tensor(): {original_tensor.size()}")
    tensor_abs_error = (interpolated_tensor - original_tensor).abs().max()
    print(f"max absolute error: {tensor_abs_error.item()}")

    # statistics
    rank = [1] + [len(Is[k][0]) for k in eqrange(1, dimension - 1)] + [1]
    original_params = 1
    for s in size:
        original_params *= s

    kept_params = 0
    for core in cores:
        kept_params += core.size(0) * core.size(1) * core.size(2)

    print("=" * 50)
    print("TCI Statistics " + "-" * 35)
    print(f"               Size: {list(size)}")
    print(f"               Rank: {rank}")
    print(f"   Parameter before: {original_params}")
    print(f"   Parameter  after: {kept_params}")
    print(f"  Compression ratio: {kept_params / original_params:.4%}")
    print(f" Max absolute error: {tensor_abs_error.item()}")
    print("=" * 50)

    return Is, Js


if __name__ == "__main__":
    dimension = 5
    size = (5, 6, 7, 8, 9)
    rank = (2, 2, 3, 4, 4)

    Us = [torch.randn(rank[i], size[i]) for i in range(dimension)]
    core = torch.randn(rank)
    tensor = torch.einsum(
        "abcde,ai,bj,ck,dl,em->ijklm",
        core,
        Us[0],
        Us[1],
        Us[2],
        Us[3],
        Us[4],
    )

    Is, Js = tensor_cross_interpolation(
        tensor=tensor, method="rook", error_threshold=1e-3
    )
