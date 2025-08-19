import math
import torch

from typing import List, Tuple, Union, Optional
from enum import Enum, auto

import time


def error_matrix_element(
    matrix: torch.Tensor,
    pivots_inverse: torch.Tensor,
    I: List[int],
    J: List[int],
    i: int,
    j: int,
) -> torch.Tensor:
    return torch.abs(matrix[i, j] - matrix[i, J] @ pivots_inverse @ matrix[I, j])


def error_matrix_row(
    matrix: torch.Tensor,
    pivots_inverse: torch.Tensor,
    I: List[int],
    J: List[int],
    i: int,
) -> torch.Tensor:
    return torch.abs(matrix[i, :] - (matrix[i, J] @ pivots_inverse) @ matrix[I, :])


def error_matrix_column(
    matrix: torch.Tensor,
    pivots_inverse: torch.Tensor,
    I: List[int],
    J: List[int],
    j: int,
) -> torch.Tensor:
    return torch.abs(matrix[:, j] - matrix[:, J] @ (pivots_inverse @ matrix[I, j]))


def error_matrix_full(
    matrix: torch.Tensor,
    pivots_inverse: torch.Tensor,
    I: List[int],
    J: List[int],
) -> torch.Tensor:
    return torch.abs(matrix - matrix[:, J] @ pivots_inverse @ matrix[I, :])


def full_search(
    matrix: torch.Tensor, pivots_inverse: torch.Tensor, I: List[int], J: List[int]
) -> Tuple[int, int, float]:

    ### This way need the whole matrix fits in memory
    error_full = error_matrix_full(matrix, pivots_inverse, I, J)
    i_star, j_star = torch.unravel_index(torch.argmax(error_full), error_full.size())
    i_star, j_star = i_star.item(), j_star.item()
    max_error = error_full[i_star, j_star].item()

    return i_star, j_star, max_error


class RookCondition(Enum):
    RULES_ROW = auto()
    RULES_COLUMN = auto()


def rook_search(
    matrix: torch.Tensor,
    pivots_inverse: torch.Tensor,
    I: List[int],
    J: List[int],
    max_iteration: int = 4,
) -> Tuple[int, int, float]:

    # pick the initial pivot
    i_star = torch.randint(0, matrix.size(0), (1,)).item()
    j_star = 0
    max_error = 0

    def rook_row():
        nonlocal i_star, j_star, max_error
        error_row = error_matrix_row(matrix, pivots_inverse, I, J, i_star)
        next_j = torch.argmax(error_row).item()

        moved = next_j != j_star
        j_star = next_j
        max_error = error_row[j_star].item()

        return moved

    def rook_column():
        nonlocal i_star, j_star, max_error
        error_column = error_matrix_column(matrix, pivots_inverse, I, J, j_star)
        next_i = torch.argmax(error_column).item()

        moved = next_i != i_star
        i_star = next_i
        max_error = error_column[i_star].item()

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

    return i_star, j_star, max_error


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
    num_cols = matrix.size(1)

    I: List[int] = []
    J: List[int] = []
    pivots_inverse = torch.inverse(matrix[I, :][:, J])

    while len(I) < num_rows and len(J) < num_cols:
        i_star, j_star, error = searcher(matrix, pivots_inverse, I, J)

        if error < error_threshold:
            break

        U_inv = pivots_inverse
        c = matrix[I, j_star][:, torch.newaxis]
        r = matrix[i_star, J][torch.newaxis, :]
        p = matrix[i_star, j_star]

        s = p - r @ U_inv @ c
        if s.abs() < 1e-2:
            # print(
            #     f"Stop due to small schur complement {s.item():.3e} at ({i_star}, {j_star})"
            # )
            break

        l = U_inv @ c
        h = r @ U_inv

        M_inv = torch.zeros(
            (len(I) + 1, len(J) + 1), dtype=matrix.dtype, device=matrix.device
        )

        M_inv = torch.vstack(
            [
                torch.hstack([U_inv + l @ (1 / s) @ h, -l @ (1 / s)]),
                torch.hstack([-(1 / s) @ h, 1 / s]),
            ]
        )

        pivots_inverse = M_inv

        I.append(i_star)
        J.append(j_star)

    return I, J, pivots_inverse
