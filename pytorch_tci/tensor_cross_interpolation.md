# Tensor cross interpolation (4d example)

**Goal** - Given a tensor $\mathcal{A} \in \mathbb{R}^{n_1 \times n_2 \times n_3 \times n_4}$ build an low-rank interpolation $\mathcal{\tilde{A}} \in \mathbb{R}^{n_1 \times n_2 \times n_3 \times n_4}$ using a small set of fibers chosen greedily.

## Notations

TBW

## Definition

For a given tensor $\mathcal{A} \in \mathbb{R}^{n_1 \times n_2 \times n_3 \times n_4}$, the tensor cross interpolation is defined as:

$$
\mathcal{A} \approx \mathcal{\tilde{A}} = \mathcal{A}(\mathbb{S}_1, J_2) \mathcal{A}(I_1, J_2)^{-1} \mathcal{A}(I_1, \mathbb{S}_2, J_3) \mathcal{A}(I_2, J_3)^{-1} \mathcal{A}(I_2, \mathbb{S}_3, J_4) \mathcal{A}(I_3, J_4)^{-1} \mathcal{A}(I_3, \mathbb{S}_4)
$$

where

$$
\begin{aligned}
    \mathbb{S}_k &= \{1, 2, ..., n_k\}, \quad k = 1, 2, 3, 4, \\
    I_1 &\subseteq \mathbb{S}_1, \\
    I_2 &\subseteq \mathbb{S}_1 \times \mathbb{S}_2, \\
    I_3 &\subseteq \mathbb{S}_1 \times \mathbb{S}_2 \times \mathbb{S}_3, \\
    J_2 &\subseteq \mathbb{S}_2 \times \mathbb{S}_3 \times \mathbb{S}_4, \\
    J_3 &\subseteq \mathbb{S}_3 \times \mathbb{S}_4, \\
    J_4 &\subseteq \mathbb{S}_4, \\
    |I_k| &= |J_k| = r.
\end{aligned}
$$

Note that, $I_k$ and $J_k$ are sets of multi-indices, and a multi-index is a tuple of indices taking care of one or more dimensions. One should understand the indexing by, for example, $I_k$ will take care of the $(1, 2, ..., k)$ dimensions, and $J_k$ will take care of the $(k, k + 1, ..., d)$ dimensions. $d = 4$ in this example.

### Interpolation
To achieve the interpolation property, that is

$$
\mathcal{A}(I_{k - 1}, \mathbb{S}_k, J_{k + 1}) = \mathcal{\tilde{A}}(I_{k - 1}, \mathbb{S}_k, J_{k + 1}), \quad k = 1, 2, 3, 4,
$$

the index sets should satisfy the nestedness condition:

$$
\begin{aligned}
    I_k &\subseteq I_{k - 1} \times \mathbb{S}_k, \\
    J_k &\subseteq \mathbb{S}_k \times J_{k + 1}.
\end{aligned}
$$

Note that, $I_0$ and $J_5$ are empty sets while the corresponding dimensions are not exist.

## Naive algorithm (left-to-right sweep DMRG greedy CI)

1. Initialize $I_k = J_k = \emptyset$, for $k = 1, 2, 3, 4$.

2. For $k$ from $1$ to $4$ do ``// (left-to-right loop over dimensions)``
   - Taking a matrix view from the tensor. That means, taking first $(k - 1)$ dimensions using $I_{k - 1}$ with a free dimension, then another free dimension and $J_{k + 1}$ for the rest dimensions. For example, when $k = 2$, we get a matrix $\mathcal{A}(I_1 \times \mathbb{S}_2, \mathbb{S}_3 \times J_4) \in \mathbb{R}^{r_1 n_2 \times n_3 r_4}$.
   - Apply ``full/rook search`` on the matrix to find a new pivot $(i_k^*, j_{k + 1}^*)$.
   - If the absolute error at position $(i_k^*, j_{k + 1}^*)$ is greater than the threshold $\epsilon$, update $I_k \leftarrow I_{k} \cup \{i_k^*\}$ and $J_{k + 1} \leftarrow J_{k + 1} \cup \{j_{k + 1}^*\}$.
3. For $k$ from $4$ to $1$ do ``// (right-to-left loop over dimensions)``
   - Do the same as step 2 but in the reverse order.

4. Repeat step 2 and 3 until no new pivots are added in the two sweeps (one left-to-right, another right-to-left).

## Implementation tips

### Query functions

- There should be query functions for element, row, and column for each superblock matrix (aka "view").
- Another approach is to query fibers. That is like querying $\mathcal{A}(I_{k - 1}, \mathbb{I}_k, J_{k + 1}) \in \mathbb{R}^{|I_{k - 1}| \times n_k \times |J_{k + 1}|}$ or $\mathcal{A}(i_{k - 1}, \mathbb{I}_k, j_{k + 1}) \in \mathbb{R}^{n_k}$.
- **For smoother implementation, I will first query superblocks in this stage.**
- Since there are CI for the superblock matrics in each inner step, the intermediate CI vectors ($e_p^{(t)}, \mathbf{e}_c^{(t)}, \mathbf{e}_r^{(t)}$) should be ephemeral and re-computed, due to the size of the view will change during the sweep.





