import torch

from typing import List, Tuple, Optional, Callable
from enum import Enum, auto


def query_error_element(
    query_matrix_element: Callable[[int, int], torch.Tensor],
    ps: torch.Tensor,  # [t, 1]
    cs: torch.Tensor,  # [t, m]
    rs: torch.Tensor,  # [t, n]
    i: int,
    j: int,
) -> torch.Tensor:
    return query_matrix_element(i, j) - ((cs[:, [i]] * rs[:, [j]]) / ps).sum()


def query_error_row(
    query_matrix_row: Callable[[int], torch.Tensor],
    ps: torch.Tensor,  # [t, 1]
    cs: torch.Tensor,  # [t, m]
    rs: torch.Tensor,  # [t, n]
    i: int,
) -> torch.Tensor:
    return query_matrix_row(i) - ((cs[:, [i]] * rs) / ps).sum(dim=0)


def query_error_column(
    query_matrix_column: Callable[[int], torch.Tensor],
    ps: torch.Tensor,  # [t, 1]
    cs: torch.Tensor,  # [t, m]
    rs: torch.Tensor,  # [t, n]
    j: int,
) -> torch.Tensor:
    return query_matrix_column(j) - ((cs * rs[:, [j]]) / ps).sum(dim=0)


def query_error_full(
    query_matrix: Callable[[], torch.Tensor],
    ps: torch.Tensor,  # [t, 1]
    cs: torch.Tensor,  # [t, m]
    rs: torch.Tensor,  # [t, n]
) -> torch.Tensor:
    return query_matrix() - (cs / ps).T @ rs


def full_search(
    query_matrix: Callable[[], torch.Tensor],
    ps: torch.Tensor,
    cs: torch.Tensor,
    rs: torch.Tensor,
) -> Tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]:

    ### This way need the whole matrix fits in memory
    error_full = query_error_full(query_matrix, ps, cs, rs)
    i_star, j_star = torch.unravel_index(
        torch.argmax(error_full.abs()), error_full.size()
    )
    i_star, j_star = i_star.item(), j_star.item()

    p = error_full[i_star, j_star]
    c = error_full[:, j_star].clone()
    r = error_full[i_star, :].clone()

    del error_full

    return i_star, j_star, p, c, r


class RookCondition(Enum):
    RULES_ROW = auto()
    RULES_COLUMN = auto()


def rook_search(
    query_matrix_element: Callable[[int, int], torch.Tensor],
    query_matrix_row: Callable[[int], torch.Tensor],
    query_matrix_column: Callable[[int], torch.Tensor],
    num_rows: int,
    ps: torch.Tensor,
    cs: torch.Tensor,
    rs: torch.Tensor,
    max_iteration: int = 4,
) -> Tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]:

    # pick the initial pivot
    i_star = torch.randint(0, num_rows, (1,)).item()
    j_star = 0
    p = None
    c = None
    r = None

    def rook_row():
        nonlocal i_star, j_star, p, c, r
        r = query_error_row(query_matrix_row, ps, cs, rs, i_star)
        next_j = torch.argmax(r.abs()).item()

        moved = next_j != j_star
        j_star = next_j
        p = r[j_star]

        return moved

    def rook_column():
        nonlocal i_star, j_star, p, c, r
        c = query_error_column(query_matrix_column, ps, cs, rs, j_star)
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

    r = query_error_row(query_matrix_row, ps, cs, rs, i_star)
    c = query_error_column(query_matrix_column, ps, cs, rs, j_star)
    p = query_error_element(query_matrix_element, ps, cs, rs, i_star, j_star)

    return i_star, j_star, p, c, r


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
) -> List[Tuple[int, int]]:
    """
    Performs matrix cross-interpolation on a given 2D tensor.

    This function implements a greedy pivot searching algorithm (full/rook) to find the optimal sets of row and column indices (I and J) for cross-interpolation, such that the approximation of the matrix based on these indices meets a specified error threshold.

    The approximation `A_tilde` is calculated as: A_tilde = A[:, J] @ inv(A[I, J]) @ A[I, :]

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
        Tuple[List[int], List[int]]: A tuple containing two lists of integers:
            - The list of selected row indices (I).
            - The list of selected column indices (J).
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

            searcher = lambda p, c, r: full_search(query_matrix, p, c, r)

        case "rook":
            assert (
                query_matrix_element is not None
            ), "query_matrix_element must be provided."
            assert query_matrix_row is not None, "query_matrix_row must be provided."
            assert (
                query_matrix_column is not None
            ), "query_matrix_column must be provided."

            searcher = lambda p, c, r: rook_search(
                query_matrix_element,
                query_matrix_row,
                query_matrix_column,
                num_rows,
                p,
                c,
                r,
            )

        case _:
            raise ValueError(f"Unknown method: {method}. Use 'full' or 'rook'.")

    element = query_matrix_element(0, 0)
    device = element.device
    dtype = element.dtype

    ps = torch.empty((0, 1), device=device, dtype=dtype)  # [t, 1]
    cs = torch.empty((0, num_rows), device=device, dtype=dtype)  # [t, m]
    rs = torch.empty((0, num_columns), device=device, dtype=dtype)  # [t, n]

    IJ = []
    while len(IJ) < min(num_rows, num_columns):
        with torch.no_grad():
            i_star, j_star, p, c, r = searcher(ps, cs, rs)

        if p.abs() < error_threshold:
            break

        if (i_star, j_star) in IJ:
            print(f"found repeated pivot ({i_star}, {j_star}) ......")
            continue

        IJ.append((i_star, j_star))

        ps = torch.cat((ps, p[torch.newaxis, torch.newaxis]), dim=0)  # [t+1, 1]
        cs = torch.cat((cs, c[torch.newaxis, :]), dim=0)  # [t+1, m]
        rs = torch.cat((rs, r[torch.newaxis, :]), dim=0)  # [t+1, n]

    I = [i for i, _ in IJ]
    J = [j for _, j in IJ]

    return I, J, None


if __name__ == "__main__":
    N, r = 600, 200
    matrix = (torch.randn(N, r) @ torch.randn(r, N)).cuda()

    I, J, _ = cross_interpolation(matrix=matrix, method="rook", error_threshold=1e-3)
    print(f"Selected rows: {I}")
    print(f"Selected cols: {J}")

    pivots_inverse = torch.linalg.inv(matrix[I, :][:, J])

    relative_error = (
        torch.norm(matrix - matrix[:, J] @ pivots_inverse @ matrix[I, :])
        / torch.norm(matrix)
    ).item()

    print(f"Relative error: {relative_error}")
