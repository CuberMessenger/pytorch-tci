# Cross interpolation

## Definition

For a given matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$, the cross interpolation is defined as:

$$
\mathbf{A} \approx \mathbf{\tilde{A}} = \mathbf{A}(\mathbb{I}, J) \mathbf{A}(I, J)^{-1} \mathbf{A}(I, \mathbb{J}),
$$

where $\mathbb{I} = \{1, 2, ..., m\}$ and $\mathbb{J} = \{1, 2, ..., n\}$, while $I \subseteq \mathbb{I}$ and $J \subseteq \mathbb{J}$ with $|I| = |J| = r$.

## Main properties

- **Interpolation**
$$
\begin{align*}
    \mathbf{A}(\mathbb{I}, J) &= \mathbf{\tilde{A}}(\mathbb{I}, J), \\
    \mathbf{A}(I, \mathbb{J}) &= \mathbf{\tilde{A}}(I, \mathbb{J}). \\
\end{align*}
$$

- **Low-rank decomposition**
$$
\mathbf{A} = \mathbf{\tilde{A}} \quad \text{if} \quad \text{rank}(\mathbf{A}) \leq r.
$$

## Algorithm

First, define an error function (matrix) as:
$$
\begin{align*}
    \mathcal{E}(i, j) &= \text{abs}(\mathbf{A} - \mathbf{\tilde{A}})(i, j) \\
    &= \text{abs}(\mathbf{A}(i, j) - \mathbf{\tilde{A}}(i, j)) \\
    &= \text{abs}(\mathbf{A}(i, j) - \mathbf{A}(i, J) \mathbf{A}(I, J)^{-1} \mathbf{A}(I, j)) \\
\end{align*}
$$
where $i \in \mathbb{I} \setminus I$ and $j \in \mathbb{J} \setminus J$.

### Full Search
Here, it first implements a ``full search`` algorithm to find the optimal sets $I$ and $J$.

Then the algorithm goes as follows:

1. Find a point $(i, j)$ that maximizes $\mathcal{E}(i, j)$:

$$
(i^*, j^*) = \argmax_{i \in \mathbb{I}, j \in \mathbb{J}} \mathcal{E}(i, j).
$$
2. Add the point $(i^*, j^*)$ to the sets $I$ and $J$:
$$
I \leftarrow I \cup \{i^*\}, \quad J \leftarrow J \cup \{j^*\}.
$$
3. Repeat steps 1 and 2 until the interpolation error satisfies a predefined threshold:
$$
\|\mathbf{\mathcal{E}}\|_{\xi} < \epsilon,
$$
where $\|\cdot\|_{\xi}$ is a norm (e.g., Frobenius norm) and $\epsilon$ is a predefined threshold.

### Rook Search
Then, implements the ``rook search`` algorithm goes as:

1. Randomly initialize a new pivot $(i^*, j^*)$.
2. Column-wise movement:

$$
i^* \leftarrow \argmax_{i \in \mathbb{I}} \mathcal{E}(i, j^*).
$$

3. Row-wise movement:

$$
j^* \leftarrow \argmax_{j \in \mathbb{J}} \mathcal{E}(i^*, j).
$$

4. Repeat steps 2 and 3 until the limit of iterations is reached or the following condition (rook condition) is met:

$$
\begin{align*}
    i^* &= \argmax_{i \in \mathbb{I}} \mathcal{E}(i, j^*), \\
    j^* &= \argmax_{j \in \mathbb{J}} \mathcal{E}(i^*, j).
\end{align*}
$$

5. Add the point $(i^*, j^*)$ to the sets $I$ and $J$:
$$
I \leftarrow I \cup \{i^*\}, \quad J \leftarrow J \cup \{j^*\}.
$$

6. Repeat steps 1 to 5 until the interpolation error satisfies a predefined threshold:
$$
\|\mathbf{\mathcal{E}}\|_{\xi} < \epsilon.
$$

## Implementation tips
Given a invertible matrix $\mathbf{U} \in \mathbb{R}^{k \times k}$ and the inverse $\mathbf{U}^{-1}$, and consider the following block matrix

$$
\mathbf{M} = \begin{pmatrix}
    \mathbf{U} & \mathbf{c} \\
    \mathbf{r} & p
\end{pmatrix} \in \mathbb{R}^{(k + 1) \times (k + 1)}
$$

where

$$
\begin{align*}
    \mathbf{c} &\in \mathbb{R}^{k \times 1}, \\
    \mathbf{r} &\in \mathbb{R}^{1 \times k}, \\
    p &\in \mathbb{R}.
\end{align*}
$$

Then the inverse of $\mathbf{M}$ can be computed as

$$
\begin{align*}
    \mathbf{M}^{-1} &= \begin{pmatrix}
        \mathbf{U}^{-1} + \mathbf{U}^{-1} \mathbf{c} s^{-1} \mathbf{r} \mathbf{U}^{-1} & -\mathbf{U}^{-1} \mathbf{c} s^{-1} \\
        -s^{-1} \mathbf{r} \mathbf{U}^{-1} & s^{-1}
    \end{pmatrix} \in \mathbb{R}^{(k + 1) \times (k + 1)} \\
    &= \begin{pmatrix}
        \mathbf{U}^{-1} + \mathbf{l} s^{-1} \mathbf{h} & -\mathbf{l} s^{-1} \\
        -s^{-1} \mathbf{h} & s^{-1}
    \end{pmatrix}
\end{align*}
$$

where

$$
\begin{align*}
    \mathbf{l} &= \mathbf{U}^{-1} \mathbf{c} \in \mathbb{R}^{k \times 1}, \\
    \mathbf{h} &= \mathbf{r} \mathbf{U}^{-1} \in \mathbb{R}^{1 \times k}, \\
    s &= p - \mathbf{r} \mathbf{U}^{-1} \mathbf{c} \in \mathbb{R}.
\end{align*}
$$

- In the above equations, $s$ is actually the Schur complement of $\mathbf{U}$ in $\mathbf{M}$, and the matrix $\mathbf{M}$ is singular when $s = 0$.
- The above equations only hold when $\mathbf{U}$ and $\mathbf{M}$ are both invertible.
