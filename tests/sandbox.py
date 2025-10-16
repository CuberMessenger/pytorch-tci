import torch


def main():
    t = torch.rand((30, 40, 10, 20), device="cuda")

    mi_1 = (1, 2, slice(None), 4)
    mi_2 = (1, slice(None), slice(None), 4)
    mi_3 = (
        torch.tensor([1, 2], dtype=torch.long, device="cuda")[:, torch.newaxis, torch.newaxis],
        torch.tensor([2, 3, 4], dtype=torch.long, device="cuda")[torch.newaxis, :, torch.newaxis],
        slice(None),
        torch.tensor([2, 1], dtype=torch.long, device="cuda")[torch.newaxis, torch.newaxis, :],
    )

    print(t[mi_1].size())
    print(t[mi_2].size())
    print(t[mi_3].size())


if __name__ == "__main__":
    main()
