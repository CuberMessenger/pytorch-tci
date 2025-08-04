import math
import torch

from typing import List, Tuple, Union, Optional
from enum import Enum, auto


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
    return torch.abs(matrix[i, :] - matrix[i, J] @ pivots_inverse @ matrix[I, :])


def error_matrix_column(
    matrix: torch.Tensor,
    pivots_inverse: torch.Tensor,
    I: List[int],
    J: List[int],
    j: int,
) -> torch.Tensor:
    return torch.abs(matrix[:, j] - matrix[:, J] @ pivots_inverse @ matrix[I, j])


def error_matrix_full(
    matrix: torch.Tensor,
    pivots_inverse: torch.Tensor,
    I: List[int],
    J: List[int],
) -> torch.Tensor:
    return torch.abs(matrix - matrix[:, J] @ pivots_inverse @ matrix[I, :])


def full_search(
    matrix: torch.Tensor, I: List[int], J: List[int]
) -> Tuple[int, int, float]:

    pivots_inverse = torch.inverse(matrix[I, :][:, J])

    error_full = error_matrix_full(matrix, pivots_inverse, I, J)

    i_star, j_star = torch.unravel_index(torch.argmax(error_full), error_full.size())

    return i_star.item(), j_star.item(), error_full[i_star, j_star].item()


class RookStatus(Enum):
    RULES = auto()
    NOT_RULES = auto()
    UNKNOWN = auto()


def random_initial_pivot(
    matrix: torch.Tensor,
    pivots_inverse: torch.Tensor,
    I: List[int],
    J: List[int],
    num_pivots: Optional[int] = None,
) -> Tuple[int, int]:
    if num_pivots is None:
        num_pivots = int(round(math.log10(matrix.numel())))

    def is_valid(i: int, j: int) -> bool:
        if i in I:
            index = I.index(i)
            if j == J[index]:
                return False
            else:
                return True
        else:
            return True

    max_error = 0.0
    i_star, j_star = 0, 0
    while num_pivots > 0:
        random_index = torch.randint(0, matrix.numel(), (1,)).item()
        i = random_index // matrix.size(1)
        j = random_index % matrix.size(1)

        if is_valid(i, j):
            error = error_matrix_element(matrix, pivots_inverse, I, J, i, j)
            if error > max_error:
                max_error = error
                i_star, j_star = i, j
            num_pivots -= 1

    return i_star, j_star


def rook_search(
    matrix: torch.Tensor, I: List[int], J: List[int], max_iteration: int = 10
) -> Tuple[int, int, float]:
    # TODO: this function need efficiency improvements

    pivots_inverse = torch.inverse(matrix[I, :][:, J])

    # pick the initial pivot
    i_star, j_star = random_initial_pivot(matrix, pivots_inverse, I, J)

    # rook search
    row_status = RookStatus.UNKNOWN  # status of the pivot in its row
    column_status = RookStatus.UNKNOWN  # status of the pivot in its column
    for index in range(max_iteration):
        if index % 2 == 0:
            # rook row-wise
            error_row = error_matrix_row(matrix, pivots_inverse, I, J, i_star)
            next_j = torch.argmax(error_row).item()

            row_status = RookStatus.RULES
            if next_j != j_star:
                column_status = RookStatus.UNKNOWN
            j_star = next_j
        else:
            # rook column-wise
            error_column = error_matrix_column(matrix, pivots_inverse, I, J, j_star)
            next_i = torch.argmax(error_column).item()

            column_status = RookStatus.RULES
            if next_i != i_star:
                row_status = RookStatus.UNKNOWN
            i_star = next_i

        if row_status is RookStatus.UNKNOWN:
            error_row = error_matrix_row(matrix, pivots_inverse, I, J, i_star)
            if torch.argmax(error_row).item() == j_star:
                row_status = RookStatus.RULES
            else:
                row_status = RookStatus.NOT_RULES

        if column_status is RookStatus.UNKNOWN:
            error_column = error_matrix_column(matrix, pivots_inverse, I, J, j_star)
            if torch.argmax(error_column).item() == i_star:
                column_status = RookStatus.RULES
            else:
                column_status = RookStatus.NOT_RULES

        if (row_status is RookStatus.RULES) and (column_status is RookStatus.RULES):
            break

    return (
        i_star,
        j_star,
        error_matrix_element(matrix, pivots_inverse, I, J, i_star, j_star).item(),
    )


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

    while len(I) < num_rows and len(J) < num_cols:
        i_star, j_star, error = searcher(matrix, I, J)

        if error < error_threshold:
            break

        I.append(i_star)
        J.append(j_star)

    return I, J
