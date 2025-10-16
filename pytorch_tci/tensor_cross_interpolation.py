import torch
from typing import List, Tuple, Callable, Optional, Union

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
) -> Tuple[
    List[int],
    List[int],
    Tuple[
        Callable[[int, int], torch.Tensor],
        Callable[[int], torch.Tensor],
        Callable[[int], torch.Tensor],
        Callable[[], torch.Tensor],
    ],
]:
    ...

def tensor_cross_interpolation(
    query_tensor_element: Optional[Callable[tuple, torch.Tensor]] = None,
    # query_matrix_row: Optional[Callable[[int], torch.Tensor]] = None,
    # query_matrix_column: Optional[Callable[[int], torch.Tensor]] = None,
    # query_matrix: Optional[Callable[[], torch.Tensor]] = None,
    # num_rows: Optional[int] = None,
    # num_columns: Optional[int] = None,
    # matrix: Optional[torch.Tensor] = None,
    # method: str = "rook",
    # error_threshold: float = 1e-3,
) -> Tuple[
    List[int],
    List[int],
    Tuple[
        Callable[[int, int], torch.Tensor],
        Callable[[int], torch.Tensor],
        Callable[[int], torch.Tensor],
        Callable[[], torch.Tensor],
    ],
]:

