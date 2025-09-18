# Tensor cross interpolation

**Let's starts with a 3D example ...**

## Definition

For a given tensor $\mathcal{A} \in \mathbb{R}^{n_1 \times n_2 \times n_3}$, the tensor cross interpolation is defined as:

$$
\mathcal{A} \approx \mathcal{\tilde{A}} = \mathcal{A}(\mathbb{I}_1, I_{\gt 1}) \mathcal{A}(I_{\leq 1}, I_{\gt 1})^{-1} \mathcal{A}(I_{\leq 1}, \mathbb{I}_2, I_{\gt 2}) \mathcal{A}(I_{\leq 2}, I_{\gt 2})^{-1} \mathcal{A}(I_{\leq 2}, \mathbb{I}_3)
$$

where

$$
\begin{aligned}
    \mathbb{I}_1 &= \{1, 2, ..., n_1\}, \\
    \mathbb{I}_2 &= \{1, 2, ..., n_2\}, \\
    \mathbb{I}_3 &= \{1, 2, ..., n_3\}, \\
    I_{\leq 1} &\subseteq \mathbb{I}_1, \\
    I_{\gt 1} &\subseteq \mathbb{I}_2 \times \mathbb{I}_3, \\
    I_{\leq 2} &\subseteq \mathbb{I}_1 \times \mathbb{I}_2, \\
    I_{\gt 2} &\subseteq \mathbb{I}_3, \\
    |I_{\leq k}| &= |I_{\gt k}| = r, \quad k = 1, 2.
\end{aligned}
$$

Note that, $I_{\leq k}$ and $I_{\gt k}$ are sets of multi-indices. One should under stand the indexing by, for example, $I_{\leq k}$ will take care of the first $k$ dimensions, and $I_{\gt k}$ will take care of the last $d - k$ dimensions.

In this case, the terms in the above interpolation can be viewed as:

| $\mathcal{A}(\mathbb{I}_1, I_{\gt 1})$ | $\mathcal{A}(I_{\leq 1}, I_{\gt 1})^{-1}$ | $\mathcal{A}(I_{\leq 1}, \mathbb{I}_2, I_{\gt 2})$ | $\mathcal{A}(I_{\leq 2}, I_{\gt 2})^{-1}$ | $\mathcal{A}(I_{\leq 2}, \mathbb{I}_3)$ |
|:-:|:-:|:-:|:-:|:-:|
| $n_1 \times r$ | $r \times r$ | $r \times n_2 \times r$ | $r \times r$ | $r \times n_3$ |

## Implementation tips

- In every step, add one new pivot for each $r_{k - 1} n_k \times n_{k + 1} r_k$ matrix.

### Tensor ACA

At step $(t - 1)$, one have

$$
\mathcal{\tilde{A}}^{(t - 1)} = \mathcal{A}(\mathbb{I}_1, I_{\gt 1}^{(t - 1)}) \mathcal{A}(I_{\leq 1}^{(t - 1)}, I_{\gt 1}^{(t - 1)})^{-1} \mathcal{A}(I_{\leq 1}^{(t - 1)}, \mathbb{I}_2, I_{\gt 2}^{(t - 1)}) \mathcal{A}(I_{\leq 2}^{(t - 1)}, I_{\gt 2}^{(t - 1)})^{-1} \mathcal{A}(I_{\leq 2}^{(t - 1)}, \mathbb{I}_3).
$$




