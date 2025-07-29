import torch

from typing import List, Tuple, Union


def ci(
    matrix: torch.Tensor, error_threshold: float = 1e-3
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

    I: List[int] = []
    J: List[int] = []

    approximation = torch.zeros_like(matrix, dtype=matrix.dtype)

    while True:
        error_matrix = (matrix - approximation).abs()

        i_star, j_star = torch.unravel_index(
            torch.argmax(error_matrix), error_matrix.size()
        )

        if error_matrix[i_star, j_star] < error_threshold:
            break

        I.append(i_star.item())
        J.append(j_star.item())

        approximation = (
            matrix[:, J] @ torch.linalg.inv(matrix[I, :][:, J]) @ matrix[I, :]
        )

    return I, J
