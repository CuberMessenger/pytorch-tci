# Tensor cross interpolation

**Goal** - Given a tensor $\mathcal{A} \in \mathbb{R}^{n_1 \times n_2 \times \cdots \times n_d}$ build an low-rank interpolation $\mathcal{\tilde{A}} \in \mathbb{R}^{n_1 \times n_2 \times \cdots \times n_d}$ using a small set of fibers chosen greedily.

## Notations

TBW

## Definition (4d example)

For a given tensor $\mathcal{A} \in \mathbb{R}^{n_1 \times n_2 \times n_3 \times n_4}$, the tensor cross interpolation is defined as:

$$
\mathcal{A} \approx \mathcal{\tilde{A}} = \mathcal{A}(\mathbb{S}_1, J_2)
\mathcal{A}(I_1, J_2)^{-1} \mathcal{A}(I_1, \mathbb{S}_2, J_3)
\mathcal{A}(I_2, J_3)^{-1} \mathcal{A}(I_2, \mathbb{S}_3, J_4)
\mathcal{A}(I_3, J_4)^{-1} \mathcal{A}(I_3, \mathbb{S}_4)
$$

where

$$
\begin{aligned}
    \mathbb{S}_k &= \{1, 2, ..., n_k\}, \\
    I_1 &\subseteq \mathbb{S}_1, \\
    I_2 &\subseteq \mathbb{S}_1 \times \mathbb{S}_2, \\
    I_3 &\subseteq \mathbb{S}_1 \times \mathbb{S}_2 \times \mathbb{S}_3, \\
    J_2 &\subseteq \mathbb{S}_2 \times \mathbb{S}_3 \times \mathbb{S}_4, \\
    J_3 &\subseteq \mathbb{S}_3 \times \mathbb{S}_4, \\
    J_4 &\subseteq \mathbb{S}_4, \\
    |I_k| &= |J_k| = r.
\end{aligned}
$$

Consequently, a single element in the interpolation can be represented as:

$$
\begin{aligned}
    \mathcal{A}(i_1, i_2, i_3, i_4) & \approx \mathcal{\tilde{A}}(i_1, i_2, i_3, i_4) \\
    &= \mathcal{A}(i_1, J_2)
    \mathcal{A}(I_1, J_2)^{-1} \mathcal{A}(I_1, i_2, J_3)
    \mathcal{A}(I_2, J_3)^{-1} \mathcal{A}(I_2, i_3, J_4)
    \mathcal{A}(I_3, J_4)^{-1} \mathcal{A}(I_3, i_4)
\end{aligned}
$$

and the supercores can be viewed as, for example:

$$
\begin{aligned}
    \mathcal{A}(I_1, \mathbb{S}_2, \mathbb{S}_3, J_4) & \approx \mathcal{\tilde{A}}(I_1, \mathbb{S}_2, \mathbb{S}_3, J_4) \\
    &= \mathcal{A}(I_1, J_2)
    \mathcal{A}(I_1, J_2)^{-1} \mathcal{A}(I_1, \mathbb{S}_2, J_3)
    \mathcal{A}(I_2, J_3)^{-1} \mathcal{A}(I_2, \mathbb{S}_3, J_4)
    \mathcal{A}(I_3, J_4)^{-1} \mathcal{A}(I_3, J_4)
\end{aligned}
$$

Note that, $I_k$ and $J_k$ are sets of multi-indices, and a multi-index is a tuple of indices taking care of one or more dimensions. One should understand the indexing by, for example, $I_k$ will take care of the $(1, 2, ..., k)$ dimensions, and $J_k$ will take care of the $(k, k + 1, ..., d)$ dimensions. $d = 4$ in this example.

## Definition

For a given tensor $\mathcal{A} \in \mathbb{R}^{n_1 \times n_2 \times \cdots \times n_d}$, the tensor cross interpolation is defined as:

$$
\mathcal{A} \approx \mathcal{\tilde{A}} = \mathcal{A}(\mathbb{S}_1, J_2)
\mathcal{A}(I_1, J_2)^{-1} \mathcal{A}(I_1, \mathbb{S}_2, J_3)
\cdots
\mathcal{A}(I_{d - 2}, \mathbb{S}_{d - 1}, J_d)
\mathcal{A}(I_{d - 1}, J_d)^{-1} \mathcal{A}(I_{d - 1}, \mathbb{S}_d)
$$

where

$$
\begin{aligned}
    \mathbb{S}_k &= \{1, 2, ..., n_k\}, \\
    I_k &\subseteq \mathbb{S}_1 \times \cdots \times \mathbb{S}_k, \quad k = 1, 2, \dots, d - 1 \\
    J_k &\subseteq \mathbb{S}_k \times \cdots \times \mathbb{S}_d, \quad k = 2, 3, \dots, d \\
    |I_k| &= |J_k| = r.
\end{aligned}
$$

Consequently, a single element in the interpolation can be represented as:

$$
\begin{aligned}
    \mathcal{A}(i_1, i_2, \dots, i_d) & \approx \mathcal{\tilde{A}}(i_1, i_2, \dots, i_d) \\
    &= \mathcal{A}(i_1, J_2)
    \mathcal{A}(I_1, J_2)^{-1} \mathcal{A}(I_1, i_2, J_3)
    \cdots
    \mathcal{A}(I_{d - 2}, i_{d - 1}, J_d)
    \mathcal{A}(I_{d - 1}, J_d)^{-1} \mathcal{A}(I_{d - 1}, i_d)
\end{aligned}
$$

and the supercores can be viewed as, for example:

$$
\begin{aligned}
    \mathcal{A}(I_{k - 1}, \mathbb{S}_k, \mathbb{S}_{k + 1}, J_{k + 2}) & \approx \mathcal{\tilde{A}}(I_{k - 1}, \mathbb{S}_k, \mathbb{S}_{k + 1}, J_{k + 2}) \\
    &= ???
\end{aligned}
$$

Note that, $I_k$ and $J_k$ are sets of multi-indices, and a multi-index is a tuple of indices taking care of one or more dimensions. One should understand the indexing by, for example, $I_k$ will take care of the $(1, 2, ..., k)$ dimensions, and $J_k$ will take care of the $(k, k + 1, ..., d)$ dimensions.

### Interpolation
To achieve the interpolation property, that is

$$
\mathcal{A}(I_{k - 1}, \mathbb{S}_k, J_{k + 1}) = \mathcal{\tilde{A}}(I_{k - 1}, \mathbb{S}_k, J_{k + 1}), \quad k = 1, 2, \dots, d,
$$

the index sets should satisfy the nestedness condition:

$$
\begin{aligned}
    I_k &\subseteq I_{k - 1} \times \mathbb{S}_k, \\
    J_k &\subseteq \mathbb{S}_k \times J_{k + 1}.
\end{aligned}
$$

Note that, $I_0$ and $J_{d + 1}$ are empty sets while the corresponding dimensions are not exist.

## DMRG (Density Matrix Renormalization Group) greedy CI (4d example)

1. Initialize $I_k$ and $J_k$, for $k = 1, 2, 3, 4$.

2. For $k$ from $1$ to $4$ do ``// (left-to-right loop over dimensions)``
   - Taking a matrix view from the tensor. That means, taking first $(k - 1)$ dimensions using $I_{k - 1}$ with a free dimension, then another free dimension and $J_{k + 1}$ for the rest dimensions. For example, when $k = 2$, we get a matrix $\mathcal{A}(I_1 \times \mathbb{S}_2, \mathbb{S}_3 \times J_4) \in \mathbb{R}^{r_1 n_2 \times n_3 r_4}$.
   - Apply ``full/rook search`` on the matrix to find a new pivot $(i_k^*, j_{k + 1}^*)$.
   - If the absolute error at position $(i_k^*, j_{k + 1}^*)$ is greater than the threshold $\epsilon$, update $I_k \leftarrow I_{k} \cup \{i_k^*\}$ and $J_{k + 1} \leftarrow J_{k + 1} \cup \{j_{k + 1}^*\}$.
3. For $k$ from $4$ to $1$ do ``// (right-to-left loop over dimensions)``
   - Do the same as step 2 but in the reverse order.

4. Repeat step 2 and 3 until no new pivots are added in the two sweeps (one left-to-right, another right-to-left).

## Implementation tips

### Initialization


- The sets need to be initialized satisfying the nestedness condition. A simple way is to set (for all valid $k$)

$$
I_k = \{(1, 1, ..., 1)\}, J_k = \{(1, 1, ..., 1)\}.
$$

- The cores (or glue matrices) are $\mathcal{A}_k = \mathcal{A}(I_k, J_{k + 1}) \in \mathbb{R}^{r_k \times r_{k + 1}}$. Then some building blocks $C_k(i_k) = \mathcal{A}(I_{k - 1}, i_k, J_{k + 1})$. They are identically, the terms in the interpolation formula. And the inverse and multiplications operations are done explicitly. It matches the complexity of the algorithm.

<!-- - For now, I will find out the vectorized version of the above operations. Then implement the algorithm, and at last optimize the performance later, maybe with ACA. -->

- When accessing the supercores, one can represent an element in a supercore using a similar formula as the interpolation one. And before high-frequency access, cache the left/right parts.

- When a new pivot is selected, it is guaranteed to satisfy the nestedness condition. And one can update only the affected parts. For example, say $\mathcal{A} \in \mathbb{R}^{3 \times 4 \times 3}$, and it has

$$
\begin{aligned}
    I_1 &= \{1, 3\}, \\
    J_2 &= \{(2, 2), (4, 3)\} \text{// pairs with } J_3, \\
    I_2 &= \{(1, 2), (3, 1)\} \text{// pairs with } I_1, \\
    J_3 &= \{2, 3\}.
\end{aligned}
$$

Then, suppose a new pivot $(i_1^*, i_2^*, i_3^*) = (2, 1, 3)$ is selected (somehow mapped to the $n$ dimensional format). Then, we only need to update

$$
\begin{aligned}
    I_1 &\leftarrow I_1 \cup \{2\} = \{1, 2, 3\}, \\
    J_2 &\leftarrow J_2 \cup \{(1, 3)\} = \{(2, 2), (4, 3), (1, 3)\}.
\end{aligned}
$$

