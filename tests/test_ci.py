import torch
from pytorch_tci import ci


def main():
    matrix = torch.rand((10, 3)) @ torch.rand((3, 10))
    I, J = ci(matrix, error_threshold=1e-3)
    print("Selected row indices:", I)
    print("Selected column indices:", J)

    error = torch.norm(
        matrix - matrix[:, J] @ torch.linalg.inv(matrix[I, :][:, J]) @ matrix[I, :]
    )
    print("Approximation error:", error.item())


if __name__ == "__main__":
    main()
