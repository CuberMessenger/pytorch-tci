import tqdm
import time
import torch
from pytorch_tci import ci


def prepare_test_matrix(N, r):
    return torch.rand((N, r)) @ torch.rand((r, N))


def test_ci_single(matrix, method):
    torch.cuda.synchronize()
    start_time = time.perf_counter_ns()
    I, J, pivots_inverse_schur = ci(matrix, method=method)
    torch.cuda.synchronize()
    end_time = time.perf_counter_ns()

    time_cost = (end_time - start_time) / 1e6  # ms

    try:
        pivots_inverse = torch.linalg.inv(matrix[I, :][:, J])
        inverse_difference = torch.norm(pivots_inverse_schur - pivots_inverse).item()
    except Exception:
        pivots_inverse = pivots_inverse_schur
        inverse_difference = None

    relative_error = (
        torch.norm(matrix - matrix[:, J] @ pivots_inverse @ matrix[I, :])
        / torch.norm(matrix)
    ).item()

    result = {
        "num_pivots": len(I),
        "I": I,
        "J": J,
        "relative_error": relative_error,
        "inverse_difference": inverse_difference,
        "time_cost": time_cost,
    }

    return result


def test_ci(N, r, method, num_iterations):
    results = []

    # warm up
    for _ in tqdm.tqdm(range(3), desc=f"Warm up {method} ......"):
        matrix = prepare_test_matrix(N, r).cuda()
        _ = test_ci_single(matrix, method)

    for _ in tqdm.tqdm(range(num_iterations), desc=f"Testing {method} ......"):
        matrix = prepare_test_matrix(N, r).cuda()
        result = test_ci_single(matrix, method)
        results.append(result)

    relative_errors = torch.Tensor([r["relative_error"] for r in results])
    time_costs = torch.Tensor([r["time_cost"] for r in results])

    print(f"Results of testing {method} for {num_iterations} iterations:")
    print(
        f"Relative error: {relative_errors.mean().item()} ± {relative_errors.std().item()}"
    )
    print(f"Time cost: {time_costs.mean().item()} ± {time_costs.std().item()}")
    print(f"")


def main():
    N, r = 120, 20
    N, r = 240, 60

    test_ci(N, r, method="full", num_iterations=16)
    test_ci(N, r, method="rook", num_iterations=16)

    """
    N, r = 240, 60
    
    Results of testing full for 16 iterations:
    Relative error: 1.6688509276718833e-05 ± 3.640356453615823e-06
    Time cost: 110.21920776367188 ± 5.356021881103516

    []
    Results of testing rook for 16 iterations:
    Relative error: 1.765836168488022e-05 ± 5.851404239365365e-06
    Time cost: 399.16705322265625 ± 32.34873962402344
    """


if __name__ == "__main__":
    main()
