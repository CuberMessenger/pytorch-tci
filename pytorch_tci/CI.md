# Cross interpolation

## Definition

For a given matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$, the cross interpolation is defined as:

$$
\mathbf{A} \approx \mathbf{\tilde{A}} = \mathbf{A}\left(\mathbb{I}, J\right) \mathbf{A}\left(I, J\right) \mathbf{A}\left(I, \mathbb{J}\right),
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
Here, it first implements a ``full search`` algorithm to find the optimal sets $I$ and $J$.

First, define an error function (matrix) as:
$$
\mathcal{E}(i, j) = |\mathbf{A} - \mathbf{\tilde{A}}|(i, j)
$$
where $i \in \mathbb{I} \setminus I$ and $j \in \mathbb{J} \setminus J$.

Then the algorithm goes as follows:

1. Find a point $(i, j)$ that maximizes $\mathcal{E}(i, j)$:

$$
i^*, j^* = \argmax_{i \in \mathbb{I} \setminus I, j \in \mathbb{J} \setminus J} \mathcal{E}(i, j).
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

