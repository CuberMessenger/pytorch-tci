import time
import torch
from pytorch_tci import ci


def prepare_test_matrix(N, r):
    return torch.rand((N, r)) @ torch.rand((r, N))


def test_ci(matrix, method):
    start_time = time.perf_counter_ns()
    I, J, pivots_inverse = ci(matrix, method=method, error_threshold=1e-4)
    end_time = time.perf_counter_ns()

    time_cost = (end_time - start_time) / 1e6  # ms

    relative_error = torch.norm(
        matrix - matrix[:, J] @ torch.linalg.inv(matrix[I, :][:, J]) @ matrix[I, :]
    ) / torch.norm(matrix)
    # relative_error = torch.norm(
    #     matrix - matrix[:, J] @ pivots_inverse @ matrix[I, :]
    # ) / torch.norm(matrix)

    inv_diff = torch.norm(torch.linalg.inv(matrix[I, :][:, J]) - pivots_inverse)

    print("Num of pivots:", len(I))
    print("Selected row indices:", I[:5], "..." if len(I) > 5 else "")
    print("Selected column indices:", J[:5], "..." if len(J) > 5 else "")
    print("Relative approximation error:", relative_error.item())
    print("Inverse difference:", inv_diff.item())
    print("Time cost:", time_cost, "ms")


def main():
    N = 60
    r = 20
    matrix = prepare_test_matrix(N, r)
    matrix = matrix.cuda()
    # print("Testing CI with full search method:")
    # test_ci(matrix, method="full")

    print("\nTesting CI with rook search method:")
    test_ci(matrix, method="rook")


if __name__ == "__main__":
    main()
