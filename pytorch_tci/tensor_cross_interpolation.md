# Tensor cross interpolation (3d example)

**Goal** - Given a tensor $\mathcal{A} \in \mathbb{R}^{n_1 \times n_2 \times n_3}$ build an low-rank interpolation $\mathcal{\tilde{A}} \in \mathbb{R}^{n_1 \times n_2 \times n_3}$ using a small set of fibers chosen greedily.

## Notations

TBW

## Definition

For a given tensor $\mathcal{A} \in \mathbb{R}^{n_1 \times n_2 \times n_3}$, the tensor cross interpolation is defined as:

$$
\mathcal{A} \approx \mathcal{\tilde{A}} = \mathcal{A}(\mathbb{I}_1, I_{\gt 1}) \mathcal{A}(I_{\leq 1}, I_{\gt 1})^{-1} \mathcal{A}(I_{\leq 1}, \mathbb{I}_2, I_{\gt 2}) \mathcal{A}(I_{\leq 2}, I_{\gt 2})^{-1} \mathcal{A}(I_{\leq 2}, \mathbb{I}_3)
$$

where $\mathbb{I} = \{1, 2, ..., m\}$ and $\mathbb{J} = \{1, 2, ..., n\}$, while $I \subseteq \mathbb{I}$ and $J \subseteq \mathbb{J}$ with $|I| = |J| = r$.

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

