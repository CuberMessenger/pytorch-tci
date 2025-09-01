import torch

from typing import List, Tuple
from enum import Enum, auto


def query_matrix_element(matrix: torch.Tensor, i: int, j: int) -> torch.Tensor:
    return matrix[i, j]


def query_matrix_row(matrix: torch.Tensor, i: int) -> torch.Tensor:
    return matrix[i, :]


def query_matrix_column(matrix: torch.Tensor, j: int) -> torch.Tensor:
    return matrix[:, j]


def query_error_element(
    matrix: torch.Tensor,
    ps: torch.Tensor,  # [t, 1]
    cs: torch.Tensor,  # [t, m]
    rs: torch.Tensor,  # [t, n]
    i: int,
    j: int,
) -> torch.Tensor:
    return query_matrix_element(matrix, i, j) - ((cs[:, [i]] * rs[:, [j]]) / ps).sum()


def query_error_row(
    matrix: torch.Tensor,
    ps: torch.Tensor,  # [t, 1]
    cs: torch.Tensor,  # [t, m]
    rs: torch.Tensor,  # [t, n]
    i: int,
) -> torch.Tensor:
    return query_matrix_row(matrix, i) - ((cs[:, [i]] * rs) / ps).sum(dim=0)


def query_error_column(
    matrix: torch.Tensor,
    ps: torch.Tensor,  # [t, 1]
    cs: torch.Tensor,  # [t, m]
    rs: torch.Tensor,  # [t, n]
    j: int,
) -> torch.Tensor:
    return query_matrix_column(matrix, j) - ((cs * rs[:, [j]]) / ps).sum(dim=0)


def query_error_full(
    matrix: torch.Tensor,
    ps: torch.Tensor,  # [t, 1]
    cs: torch.Tensor,  # [t, m]
    rs: torch.Tensor,  # [t, n]
) -> torch.Tensor:
    return matrix - (
        torch.bmm(cs[:, :, torch.newaxis], rs[:, torch.newaxis, :])
        / ps[:, :, torch.newaxis]
    ).sum(dim=0)


def full_search(
    matrix: torch.Tensor, ps: torch.Tensor, cs: torch.Tensor, rs: torch.Tensor
) -> Tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]:

    ### This way need the whole matrix fits in memory
    error_full = query_error_full(matrix, ps, cs, rs)
    i_star, j_star = torch.unravel_index(
        torch.argmax(error_full.abs()), error_full.size()
    )
    i_star, j_star = i_star.item(), j_star.item()

    p = error_full[i_star, j_star]
    c = error_full[:, j_star]
    r = error_full[i_star, :]

    return i_star, j_star, p, c, r


class RookCondition(Enum):
    RULES_ROW = auto()
    RULES_COLUMN = auto()


def rook_search(
    matrix: torch.Tensor,
    ps: torch.Tensor,
    cs: torch.Tensor,
    rs: torch.Tensor,
    I: List[int] = [],
    max_iteration: int = 4,
) -> Tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]:

    # pick the initial pivot
    i_star = torch.randint(0, matrix.size(0), (1,)).item()
    while i_star in I:
        i_star = torch.randint(0, matrix.size(0), (1,)).item()
    j_star = 0
    p = None
    c = None
    r = None

    def rook_row():
        nonlocal i_star, j_star, p, c, r
        r = query_error_row(matrix, ps, cs, rs, i_star)
        next_j = torch.argmax(r.abs()).item()

        moved = next_j != j_star
        j_star = next_j
        p = r[j_star]

        return moved

    def rook_column():
        nonlocal i_star, j_star, p, c, r
        c = query_error_column(matrix, ps, cs, rs, j_star)
        next_i = torch.argmax(c.abs()).item()

        moved = next_i != i_star
        i_star = next_i
        p = c[i_star]

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

    return i_star, j_star, p, c, r


def ci(
    matrix: torch.Tensor, method: str = "rook", error_threshold: float = 1e-3
) -> Tuple[List[int], List[int]]:
    """
    Performs matrix cross-interpolation on a given 2D tensor.

    This function implements a full search algorithm to find the optimal sets
    of row and column indices (I and J) for cross-interpolation, such that
    the approximation of the matrix based on these indices meets a specified
    error threshold.

    The approximation `A_tilde` is calculated as:
    A_tilde = A[:, J] @ inv(A[I, J]) @ A[I, :]

    Args:
        matrix (torch.Tensor): The 2D input matrix to interpolate.
        method (str): The method to use for finding the indices. Available options are:
            - "full": Uses a full search algorithm to find the optimal indices.
            - "rook": Uses a rook search algorithm to find the optimal indices.
        error_threshold (float): The desired precision. The algorithm stops
            when the Frobenius norm of the error matrix (A - A_tilde) is
            below this threshold.

    Returns:
        Tuple[List[int], List[int]]: A tuple containing two lists of integers:
            - The list of selected row indices (I).
            - The list of selected column indices (J).
    """
    if matrix.dim() != 2:
        raise ValueError("Input matrix must be a 2D tensor.")

    match method:
        case "full":
            searcher = full_search
        case "rook":
            searcher = rook_search
        case _:
            raise ValueError(f"Unknown method: {method}. Use 'full' or 'rook'.")

    num_rows = matrix.size(0)
    num_columns = matrix.size(1)

    I: List[int] = []
    J: List[int] = []

    ps = torch.empty((0, 1), device=matrix.device, dtype=matrix.dtype)  # [t, 1]
    cs = torch.empty((0, num_rows), device=matrix.device, dtype=matrix.dtype)  # [t, m]
    rs = torch.empty(
        (0, num_columns), device=matrix.device, dtype=matrix.dtype
    )  # [t, n]

    while len(I) < num_rows and len(J) < num_columns:
        i_star, j_star, p, c, r = searcher(matrix, ps, cs, rs, I=I)

        if p.abs() < error_threshold:
            break

        I.append(i_star)
        J.append(j_star)

        ps = torch.cat((ps, p[torch.newaxis, torch.newaxis]), dim=0)  # [t+1, 1]
        cs = torch.cat((cs, c[torch.newaxis, :]), dim=0)  # [t+1, m]
        rs = torch.cat((rs, r[torch.newaxis, :]), dim=0)  # [t+1, n]

    return I, J, None
