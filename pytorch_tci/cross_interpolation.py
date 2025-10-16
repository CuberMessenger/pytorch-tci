import torch

from typing import Optional, Callable
from enum import Enum, auto


def query_interpolation_element(
    eps: torch.Tensor,  # [t, 1]
    ecs: torch.Tensor,  # [t, m]
    ers: torch.Tensor,  # [t, n]
    i: int,
    j: int,
) -> torch.Tensor:
    return ((ecs[:, [i]] * ers[:, [j]]) / eps).sum()


def query_interpolation_column(
    eps: torch.Tensor,  # [t, 1]
    ecs: torch.Tensor,  # [t, m]
    ers: torch.Tensor,  # [t, n]
    j: int,
) -> torch.Tensor:
    return ((ecs * ers[:, [j]]) / eps).sum(dim=0)


def query_interpolation_row(
    eps: torch.Tensor,  # [t, 1]
    ecs: torch.Tensor,  # [t, m]
    ers: torch.Tensor,  # [t, n]
    i: int,
) -> torch.Tensor:
    return ((ecs[:, [i]] * ers) / eps).sum(dim=0)


def query_interpolation_full(
    eps: torch.Tensor,  # [t, 1]
    ecs: torch.Tensor,  # [t, m]
    ers: torch.Tensor,  # [t, n]
) -> torch.Tensor:
    return (ecs / eps).T @ ers


def query_error_element(
    query_matrix_element: Callable[[int, int], torch.Tensor],
    eps: torch.Tensor,  # [t, 1]
    ecs: torch.Tensor,  # [t, m]
    ers: torch.Tensor,  # [t, n]
    i: int,
    j: int,
) -> torch.Tensor:
    return query_matrix_element(i, j) - query_interpolation_element(eps, ecs, ers, i, j)


def query_error_column(
    query_matrix_column: Callable[[int], torch.Tensor],
    eps: torch.Tensor,  # [t, 1]
    ecs: torch.Tensor,  # [t, m]
    ers: torch.Tensor,  # [t, n]
    j: int,
) -> torch.Tensor:
    return query_matrix_column(j) - query_interpolation_column(eps, ecs, ers, j)


def query_error_row(
    query_matrix_row: Callable[[int], torch.Tensor],
    eps: torch.Tensor,  # [t, 1]
    ecs: torch.Tensor,  # [t, m]
    ers: torch.Tensor,  # [t, n]
    i: int,
) -> torch.Tensor:
    return query_matrix_row(i) - query_interpolation_row(eps, ecs, ers, i)


def query_error_full(
    query_matrix: Callable[[], torch.Tensor],
    eps: torch.Tensor,  # [t, 1]
    ecs: torch.Tensor,  # [t, m]
    ers: torch.Tensor,  # [t, n]
) -> torch.Tensor:
    return query_matrix() - query_interpolation_full(eps, ecs, ers)


def full_search(
    query_matrix: Callable[[], torch.Tensor],
    eps: torch.Tensor,
    ecs: torch.Tensor,
    ers: torch.Tensor,
) -> tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]:

    ### This way need the whole matrix fits in memory
    error_full = query_error_full(query_matrix, eps, ecs, ers)
    i_star, j_star = torch.unravel_index(
        torch.argmax(error_full.abs()), error_full.size()
    )
    i_star, j_star = i_star.item(), j_star.item()

    ep = error_full[i_star, j_star]
    ec = error_full[:, j_star].clone()
    er = error_full[i_star, :].clone()

    del error_full

    return i_star, j_star, ep, ec, er


class RookCondition(Enum):
    RULES_ROW = auto()
    RULES_COLUMN = auto()


def rook_search(
    query_matrix_element: Callable[[int, int], torch.Tensor],
    query_matrix_row: Callable[[int], torch.Tensor],
    query_matrix_column: Callable[[int], torch.Tensor],
    eps: torch.Tensor,
    ecs: torch.Tensor,
    ers: torch.Tensor,
    max_iteration: int = 4,
) -> tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]:

    # pick the initial pivot
    i_star = torch.randint(0, ecs.size(1), (1,)).item()
    j_star = 0
    ep = None
    ec = None
    er = None

    def rook_row():
        nonlocal i_star, j_star, ep, ec, er
        er = query_error_row(query_matrix_row, eps, ecs, ers, i_star)
        next_j = torch.argmax(er.abs()).item()

        moved = next_j != j_star
        j_star = next_j
        ep = er[j_star]

        return moved

    def rook_column():
        nonlocal i_star, j_star, ep, ec, er
        ec = query_error_column(query_matrix_column, eps, ecs, ers, j_star)
        next_i = torch.argmax(ec.abs()).item()

        moved = next_i != i_star
        i_star = next_i
        ep = ec[i_star]

        return moved

    # rook search
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

    er = query_error_row(query_matrix_row, eps, ecs, ers, i_star)
    ec = query_error_column(query_matrix_column, eps, ecs, ers, j_star)
    ep = query_error_element(query_matrix_element, eps, ecs, ers, i_star, j_star)

    return i_star, j_star, ep, ec, er


def cross_interpolation(
    query_matrix_element: Optional[Callable[[int, int], torch.Tensor]] = None,
    query_matrix_row: Optional[Callable[[int], torch.Tensor]] = None,
    query_matrix_column: Optional[Callable[[int], torch.Tensor]] = None,
    query_matrix: Optional[Callable[[], torch.Tensor]] = None,
    num_rows: Optional[int] = None,
    num_columns: Optional[int] = None,
    matrix: Optional[torch.Tensor] = None,
    method: str = "rook",
    error_threshold: float = 1e-3,
) -> tuple[
    list[int],
    list[int],
    tuple[
        Callable[[int, int], torch.Tensor],
        Callable[[int], torch.Tensor],
        Callable[[int], torch.Tensor],
        Callable[[], torch.Tensor],
    ],
]:
    """
    Performs matrix cross-interpolation on a given 2D torch.Tensor.

    This function implements a greedy pivot searching algorithm (full/rook) to find the optimal sets of row and column indices (I and J) for cross-interpolation, such that the interpolation of the matrix based on these indices meets a specified error threshold.

    The interpolation `A_tilde` is calculated as A_tilde = A[:, J] @ inv(A[I, J]) @ A[I, :]. However, the function does not return the interpolation directly. Instead, it returns the anonymous functions for computing the interpolation (fully or at given locations).

    Arguments:
        query_matrix_element: A callable that takes two integers (i, j) and returns the element at position (i, j) of the matrix.
        query_matrix_row: A callable that takes an integer i and returns the i-th row of the matrix.
        query_matrix_column: A callable that takes an integer j and returns the j-th column of the matrix.
        query_matrix: A callable that returns the full matrix as a 2D tensor.
        num_rows: The number of rows in the matrix. Required if `matrix` is not provided.
        num_columns: The number of columns in the matrix. Required if `matrix` is not provided.
        matrix: A shorthand way to provide the full matrix as a 2D tensor. If provided, this will be used instead of the query functions.
        method: The method to use for finding the indices. Available options are:
            - "full": Uses a full search algorithm to find the optimal indices.
            - "rook": Uses a rook search algorithm to find the optimal indices.
        error_threshold: The desired precision. The algorithm stops when the new pivot's error is below this threshold.

    Returns:
        tuple[list[int], list[int], tuple[
            Callable[[int, int], torch.Tensor],
            Callable[[int], torch.Tensor],
            Callable[[int], torch.Tensor],
            Callable[[], torch.Tensor],
        ]]:
        A tuple containing:
            - I: A list of selected row indices.
            - J: A list of selected column indices.
            - (query_interpolation_element, query_interpolation_row, query_interpolation_column, query_interpolation_full):
              A tuple of callables for computing the interpolation based on the selected indices.
    """
    if matrix is not None:
        assert matrix.dim() == 2, "Input matrix must be a 2D tensor."
        query_matrix_element = lambda i, j: matrix[i, j]
        query_matrix_row = lambda i: matrix[i, :]
        query_matrix_column = lambda j: matrix[:, j]
        query_matrix = lambda: matrix
        num_rows, num_columns = matrix.size()

    assert num_rows is not None, "num_rows must be provided."
    assert num_columns is not None, "num_columns must be provided."

    match method:
        case "full":
            assert query_matrix is not None, "query_matrix must be provided."

            searcher = lambda args: full_search(query_matrix, *args)

        case "rook":
            assert (
                query_matrix_element is not None
            ), "query_matrix_element must be provided."
            assert query_matrix_row is not None, "query_matrix_row must be provided."
            assert (
                query_matrix_column is not None
            ), "query_matrix_column must be provided."

            searcher = lambda args: rook_search(
                query_matrix_element, query_matrix_row, query_matrix_column, *args
            )

        case _:
            raise ValueError(f"Unknown method: {method}. Use 'full' or 'rook'.")

    element = query_matrix_element(0, 0)
    device = element.device
    dtype = element.dtype

    eps = torch.empty((0, 1), device=device, dtype=dtype)  # [t, 1]
    ecs = torch.empty((0, num_rows), device=device, dtype=dtype)  # [t, m]
    ers = torch.empty((0, num_columns), device=device, dtype=dtype)  # [t, n]

    I: list[int] = []
    J: list[int] = []
    pivots = set()

    while len(pivots) < min(num_rows, num_columns):
        with torch.no_grad():
            i_star, j_star, ep, ec, er = searcher((eps, ecs, ers))

        if ep.abs() < error_threshold:
            break

        if (i_star, j_star) in pivots:
            print(f"found repeated pivot ({i_star}, {j_star}) ......")
            continue

        I.append(i_star)
        J.append(j_star)
        pivots.add((i_star, j_star))

        eps = torch.cat((eps, ep[torch.newaxis, torch.newaxis]), dim=0)  # [t+1, 1]
        ecs = torch.cat((ecs, ec[torch.newaxis, :]), dim=0)  # [t+1, m]
        ers = torch.cat((ers, er[torch.newaxis, :]), dim=0)  # [t+1, n]

    return (
        I,
        J,
        (
            lambda i, j: query_interpolation_element(eps, ecs, ers, i, j),
            lambda i: query_interpolation_row(eps, ecs, ers, i),
            lambda j: query_interpolation_column(eps, ecs, ers, j),
            lambda: query_interpolation_full(eps, ecs, ers),
        ),
    )


if __name__ == "__main__":
    M, r, N = 500, 80, 300
    matrix = (torch.randn(M, r) @ torch.randn(r, N)).cuda()

    I, J, _ = cross_interpolation(matrix=matrix, method="rook", error_threshold=1e-3)

    # import matplotlib.pyplot as plot

    # figure, axis = plot.subplots(1, 2, dpi=200, figsize=(12, 6))

    # x = torch.arange(1, len(relative_errors) + 1)
    # y = torch.tensor(relative_errors) * 100

    # axis[0].plot(x, y, marker="o")
    # axis[0].set_xlabel("Iteration", fontsize=12)
    # axis[0].set_ylabel("Relative Error (%)", fontsize=12)
    # axis[0].set_title(f"Relative Error with N={N}, r={r}", fontsize=12)

    # x = torch.arange(1, len(num_singulars) + 1)
    # y = torch.tensor(num_singulars)
    # axis[1].plot(x, y, marker="o", color="orange")
    # axis[1].set_xlabel("Iteration", fontsize=12)
    # axis[1].set_ylabel("# Singular Values", fontsize=12)
    # axis[1].set_title(f"# Singular Values in E with N={N}, r={r}", fontsize=12)

    # plot.show()
