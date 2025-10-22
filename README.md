# Description

This repository implements **Tensor Cross Interpolation** algorithm upon the **PyTorch** framework. Targeting at expanding the applicability of the algorithm in recent deep learning tasks.


# References
- Y. N. Fernández et al., “Learning Feynman Diagrams with Tensor Trains,” 2022.


# TODO list
- [x] Implement the **Cross Interpolation** algorithm without memory limit.
- [ ] Implement the **Tensor Cross Interpolation** algorithm without memory limit.
- [ ] Implement the **Cross Interpolation** algorithm with memory limit.
- [ ] Implement the **Tensor Cross Interpolation** algorithm with memory limit.
- [ ] Revise the implementation with partial rank-revealing LU decomposition for better numerical stability.

# Notes
- With a little pre-experiments, it seems rook search cost less runtime memory even when the full tensor fits in memory.
- The 2014-Quasioptimality paper is a better reference.
