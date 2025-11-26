import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function


class LasorLinear(nn.Linear):
    def forward(self, input):
        raise NotImplementedError("LasorLinear is not implemented yet.")


def main():
    repeat = 30

    model = nn.Sequential(
        nn.Linear(100, 5120),
        nn.ReLU(),
        nn.Linear(5120, 10240),
        nn.ReLU(),
        nn.Linear(10240, 5120),
        nn.ReLU(),
        nn.Linear(5120, 10),
    )

    inputs = torch.rand((1024, 100))
    ground_truth = torch.rand((1024, 10))

    model, inputs = model.cuda(), inputs.cuda()
    ground_truth = ground_truth.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.MSELoss()

    ### Profiler
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #     for _ in range(repeat):
    #         # with record_function("forward"):
    #         outputs = model(inputs)

    #         loss = loss_function(outputs, ground_truth)

    #         optimizer.zero_grad()
    #         # with record_function("backward"):
    #         loss.backward()
    #         optimizer.step()

    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))

    ### CUDA event
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    forward_time_ms = 0.0
    backward_time_ms = 0.0
    for _ in range(repeat):
        torch.cuda.synchronize()
        starter.record()
        outputs = model(inputs)
        ender.record()
        torch.cuda.synchronize()

        forward_time_ms += starter.elapsed_time(ender)

        loss = loss_function(outputs, ground_truth)
        optimizer.zero_grad()

        torch.cuda.synchronize()
        starter.record()
        loss.backward()
        ender.record()
        torch.cuda.synchronize()

        backward_time_ms += starter.elapsed_time(ender)

        optimizer.step()

    print(f"Average forward time: {forward_time_ms / repeat:.3f} ms")
    print(f"Average backward time: {backward_time_ms / repeat:.3f} ms")

if __name__ == "__main__":
    main()

