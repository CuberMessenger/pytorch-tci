import torch

from typing import Callable, Optional

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


def query_tensor_superblock(
    Is: MultiIndex,
    Js: MultiIndex,
    query_tensor_element: Callable[[MultiIndex], torch.Tensor] = None,
    query_tensor_element_batched: Callable[[BatchedMultiIndex], torch.Tensor] = None,
):
    if query_tensor_element_batched is None:
        raise NotImplementedError("Non-batched mode is not implemented yet.")
    
    # the dimensions taking cared by Is and Js are always moved to the front
    if len(Is) == 0:
        raise NotImplementedError("Special case of empty Is is not implemented yet.")

    if len(Js) == 0:
        raise NotImplementedError("Special case of empty Js is not implemented yet.")
    
    superblock = query_tensor_element_batched(Is + (slice(None), slice(None)) + Js).permute((0, 2, 3, 1))

    return superblock    

def tensor_cross_interpolation(
    query_tensor_element: Optional[Callable[[MultiIndex], torch.Tensor]] = None,
    query_tensor_element_batched: Optional[Callable[[BatchedMultiIndex], torch.Tensor]] = None,
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
            ...

        case "rook":
            ...

        case _:
            raise ValueError(f"Unknown method: {method}")

    element = query_tensor_element((0,) * dimension)
    device = element.device
    dtype = element.dtype

    ## A MultiIndex is a tuple, whose elements are either lists of integers or slices.
    ## Is[k] is one MultiIndex with length k
    ## Js[k] is one MultiIndex with length dimension - k + 1

    Is = [()] + [tuple([0] for _ in range(k)) for k in eqrange(1, dimension - 1)]
    #    I_0     I_1, ..., I_{d-1}
    #   Is[0]   Is[1]      Is[d-1]
    Js = (
        [None, None]
        + [tuple([0] for _ in range(dimension - k + 1)) for k in eqrange(2, dimension)]
        + [()]
    )
    #                    J_2, ..., J_d                                            J_{d+1}
    #                   Js[2]      Js[d]                                          Js[d+1]

    # """
    ### EXAMPLE ###

    Is[0] = ()
    Is[1] = ([[2], [0]],)  # (2, ...) and (0, ...)
    Is[2] = ([[2], [2]], [[0], [1]])  # (2, 0, ...) and (2, 1, ...)
    Is[3] = ([[2], [2]], [[0], [1]], [[3], [4]])  # (2, 0, 3, ...) and (2, 1, 4, ...)
    Is[4] = ([[2], [2]], [[0], [1]], [[3], [4]], [[0], [5]])  # (2, 0, 3, 0) and (2, 1, 4, 5)

    Js[0] = None
    Js[1] = None
    Js[2] = ([[1, 2]], [[2, 0]], [[3, 3]], [[1, 1]])  # (..., 1, 2, 3, 1) and (..., 2, 0, 3, 1)
    Js[3] = ([[2, 0]], [[3, 3]], [[1, 1]])  # (..., 2, 3, 1) and (..., 0, 3, 1)
    Js[4] = ([[3, 4]], [[1, 1]])  # (..., 3, 1) and (..., 4, 1)
    Js[5] = ([[1, 4, 5]],)  # (..., 1) and (..., 4) and (..., 5)
    Js[6] = ()

    ### EXAMPLE ###
    # """

    while True:
        # left-to-right sweep
        for k in eqrange(1, dimension - 1):
            # find a new pivot
            ## form the supercore

            supercore = query_tensor_superblock(
                Is=Is[k - 1],
                Js=Js[k + 2],
                query_tensor_element=query_tensor_element,
                query_tensor_element_batched=query_tensor_element_batched,
            )
            print(f"k = {k}, supercore: {list(supercore.size())}")
            # r_{k - 1} * n_k, n_{k + 1} * r_{k + 2}

            # ## form the supercore approximation using current Is, Js
            # supercore_approx = ...

            # mi = Is[k - 1] + (slice(None),) + Js[k + 1]
            # sca_l = query_tensor_element(mi)
            # # r_{k - 1} * n_k * r_{k + 1}

            # mi = Is[k] + Js[k + 1]
            # sca_m = query_tensor_element(mi)
            # # r_{k} * r_{k + 1}

            # mi = Is[k] + (slice(None),) + Js[k + 2]
            # sca_r = query_tensor_element(mi)
            # # r_{k} * n_{k + 1} * r_{k + 2}

            # print(
            #     f"k = {k}, supercore_approx: {list(sca_l.size())} -- {list(sca_m.size())} -- {list(sca_r.size())}"
            # )

            ## apply cross interpolation to the supercore
            ...

            # update Is[k], Js[k]
            ...

        break

        # right-to-left sweep
        for k in reversed(range(1, dimension)):
            # find a new pivot
            ...

            # update Is[k], Js[k]
            ...

        # check convergence
        ...


def test_take_superblock():
    tensor = torch.randn(5, 6, 7, 8, 9)

    Is2 = ([2, 2], [0, 1])  # (2, 0, ...) and (2, 1, ...)
    Js5 = ([1, 4, 5],)  # (..., 1) and (..., 4) and (..., 5)

    ### 1
    superblock1 = torch.zeros((2, 7, 8, 3), dtype=tensor.dtype)
    for l in range(2):
        for i in range(7):
            for j in range(8):
                for r in range(3):
                    mi = (Is2[0][l], Is2[1][l], i, j, Js5[0][r])
                    superblock1[l, i, j, r] = tensor[mi]

    ### 2
    superblock2 = tensor[*Is2][..., *Js5]

    ### 3
    left = [[[2], [2]], [[0], [1]]]
    right = [[[1, 4, 5]]]

    superblock3 = tensor[tuple(left + [slice(None), slice(None)] + right)]
    superblock3 = superblock3.permute(0, 2, 3, 1)
    # this always produce 4d tensor with the dims from Is and Js being moved to the 1st and 2nd dims

    print(torch.allclose(superblock1, superblock2))
    print(torch.allclose(superblock1, superblock3))

    print(superblock1.size())


def test_take_fiber():
    tensor = torch.randn(5, 6, 7, 8, 9)

    Is2 = ([2, 2], [0, 1])  # (2, 0, ...) and (2, 1, ...)
    Js4 = (
        [2, 3, 3],
        [1, 4, 5],
    )  # (..., 2, 1) and (..., 3, 4) and (..., 3, 5)

    ### 1
    fiber1 = torch.zeros((2, 7, 3), dtype=tensor.dtype)
    for l in range(2):
        for i in range(7):
            for r in range(3):
                mi = (Is2[0][l], Is2[1][l], i, Js4[0][r], Js4[1][r])
                fiber1[l, i, r] = tensor[mi]

    ### 2
    fiber2 = tensor[*Is2][..., *Js4]

    ### 3
    left = [[[2], [2]], [[0], [1]]]
    right = [[[2, 3, 3]], [[1, 4, 5]]]

    fiber3 = tensor[tuple(left + [slice(None)] + right)]

    fiber3 = fiber3.permute(0, 2, 1)

    print(torch.allclose(fiber1, fiber2))
    print(torch.allclose(fiber1, fiber3))

    print(fiber1.size())


if __name__ == "__main__":
    tensor = torch.randn(5, 6, 7, 8, 9)
    tci = tensor_cross_interpolation(tensor=tensor, method="rook")

    # test_take_fiber()
    # test_take_superblock()
