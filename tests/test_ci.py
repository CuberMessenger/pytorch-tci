import torch
from pytorch_tci import ci


def prepare_test_matrix(N, r):
    return torch.rand((N, r)) @ torch.rand((r, N))


def test_ci(matrix, method):
    I, J = ci(matrix, method=method, error_threshold=1e-4)
    error = torch.norm(
        matrix - matrix[:, J] @ torch.linalg.inv(matrix[I, :][:, J]) @ matrix[I, :]
    )
    print("Num of pivots:", len(I))
    print("Selected row indices:", I)
    print("Selected column indices:", J)
    print("Approximation error:", error.item())


def main():
    N = 90
    r = 17
    matrix = prepare_test_matrix(N, r)
    print("Testing CI with full search method:")
    test_ci(matrix, method="full")

    print("\nTesting CI with rook search method:")
    test_ci(matrix, method="rook")


if __name__ == "__main__":
    main()
