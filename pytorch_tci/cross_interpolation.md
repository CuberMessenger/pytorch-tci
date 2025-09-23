# Cross interpolation

**Goal** - Given a matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ build an low-rank interpolation $\mathbf{\tilde{A}} \in \mathbb{R}^{m \times n}$ using a small set of rows and columns chosen greedily.

## Notations

| Symbol | Description |
|:-:|:-:|
| $\mathbf{A}$ | Original matrix |
| $\mathbf{\tilde{A}}$ | Low-rank interpolation matrix |
| $m$ | Number of rows of $\mathbf{A}$ |
| $n$ | Number of columns of $\mathbf{A}$ |
| $r$ | Rank of the interpolation |
| $\mathbb{I}$ | $\{1, 2, ..., m\}$, full row index set |
| $\mathbb{J}$ | $\{1, 2, ..., n\}$, full column index set |
| $I$ | Subset of $\mathbb{I}$, selected row indices |
| $J$ | Subset of $\mathbb{J}$, selected column indices |

## Definition

For a given matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$, the cross interpolation is defined as:

$$
\mathbf{A} \approx \mathbf{\tilde{A}} = \mathbf{A}(\mathbb{I}, J) \mathbf{A}(I, J)^{-1} \mathbf{A}(I, \mathbb{J}),
$$

where $\mathbb{I} = \{1, 2, ..., m\}$ and $\mathbb{J} = \{1, 2, ..., n\}$, while $I \subseteq \mathbb{I}$ and $J \subseteq \mathbb{J}$ with $|I| = |J| = r$.

The above definition has the following properties:

- **Interpolation**

$$
\begin{aligned}
    \mathbf{A}(\mathbb{I}, J) &= \mathbf{\tilde{A}}(\mathbb{I}, J), \\
    \mathbf{A}(I, \mathbb{J}) &= \mathbf{\tilde{A}}(I, \mathbb{J}). \\
\end{aligned}
$$

- **Low-rank decomposition**

$$
\mathbf{A} = \mathbf{\tilde{A}} \quad \text{if} \quad \text{rank}(\mathbf{A}) \leq r.
$$

## Naive algorithm

First, define an error matrix as

$$
\mathcal{E}(i, j) = (\mathbf{A} - \mathbf{\tilde{A}})(i, j).
$$

Then, the algorithm goes as follows:

1. Initialize $I = J = \emptyset$.

2. Find a pivot $(i^*, j^*)$ that maximizes $\mathcal{E}(i, j)$.

3. Add the point $(i^*, j^*)$ to the sets $I$ and $J$.

4. Repeat steps 2 and 3 until $|\mathcal{E}(i^*, j^*)| \leq \epsilon$ or the maximum rank is reached.

At **step 2**, there are two strategies to find the pivot, namely **full search** and **rook search**.

### Full search

The ``full search`` strategy finds the pivot by searching through all the entries of the error matrix $\mathcal{E}$:

$$
(i^*, j^*) = \argmax_{i \in \mathbb{I}, j \in \mathbb{J}} \mathcal{E}(i, j).
$$

### Rook search

The ``rook search`` strategy finds the pivot by alternatingly searching through the rows and columns of the error matrix $\mathcal{E}$:

1. Randomly initialize a new pivot $(i^*, j^*)$.
2. Column-wise movement:

$$
i^* \leftarrow \argmax_{i \in \mathbb{I}} \mathcal{E}(i, j^*).
$$

3. Row-wise movement:

$$
j^* \leftarrow \argmax_{j \in \mathbb{J}} \mathcal{E}(i^*, j).
$$

4. Repeat steps 2 and 3 until the limit of iterations is reached or the **rook condition** is met:

$$
\begin{aligned}
    i^* &= \argmax_{i \in \mathbb{I}} \mathcal{E}(i, j^*), \\
    j^* &= \argmax_{j \in \mathbb{J}} \mathcal{E}(i^*, j).
\end{aligned}
$$

## ACA (Adaptive Cross Approximation) Algorithm

The above naive algorithm requires to compute the inverse of the pivot matrix $\mathbf{A}(I, J)$ repeatedly, which is not scalable.

The ACA algorithm addresses this issue by using an incremental update of the interpolation matrix $\mathbf{\tilde{A}}$, which avoids the need to compute the inverse of the pivot matrix from scratch at each step.

### Incremental update

At step $(t - 1)$, one has

$$
\mathbf{\tilde{A}}^{(t - 1)} = \mathbf{A}(\mathbb{I}, J^{(t - 1)}) \mathbf{A}(I^{(t - 1)}, J^{(t - 1)})^{-1} \mathbf{A}(I^{(t - 1)}, \mathbb{J})
$$

where

$$
\begin{aligned}
    I^{(t - 1)} = \{i_1, i_2, ..., i_{t - 1}\}, \\
    J^{(t - 1)} = \{j_1, j_2, ..., j_{t - 1}\}
\end{aligned}
$$

with

$$
\begin{aligned}
    \mathbf{A}(\mathbb{I}, J^{(t - 1)}) &\in \mathbb{R}^{m \times (t - 1)}, \\
    \mathbf{A}(I^{(t - 1)}, J^{(t - 1)}) &\in \mathbb{R}^{(t - 1) \times (t - 1)}, \\
    \mathbf{A}(I^{(t - 1)}, \mathbb{J}) &\in \mathbb{R}^{(t - 1) \times n}.
\end{aligned}
$$

Then, a new pivot $(i_t, j_t)$ is somehow found, which leads to the next step interpolation

$$
\mathbf{\tilde{A}}^{(t)} = \mathbf{A}(\mathbb{I}, J^{(t)}) \mathbf{A}(I^{(t)}, J^{(t)})^{-1} \mathbf{A}(I^{(t)}, \mathbb{J})
$$

where

$$
\begin{aligned}
    \mathbf{A}(\mathbb{I}, J^{(t)}) &= \begin{bmatrix}
        \mathbf{A}(\mathbb{I}, J^{(t - 1)}) & \mathbf{A}(\mathbb{I}, j_t)
    \end{bmatrix} \in \mathbb{R}^{m \times t}, \\

    \mathbf{A}(I^{(t)}, J^{(t)}) &= \begin{bmatrix}
        \mathbf{A}(I^{(t - 1)}, J^{(t - 1)}) & \mathbf{c}^{(t)} \\
        \mathbf{r}^{(t)} & \mathbf{A}(i_t, j_t)
    \end{bmatrix} \in \mathbb{R}^{t \times t}, \\

    \mathbf{A}(I^{(t)}, \mathbb{J}) &= \begin{bmatrix}
        \mathbf{A}(I^{(t - 1)}, \mathbb{J}) \\
        \mathbf{A}(i_t, \mathbb{J})
    \end{bmatrix} \in \mathbb{R}^{t \times n}.
\end{aligned}
$$

Here, one can use a technique called **Schur complement** to represent the inverse of the new pivot matrix $\mathbf{A}(I^{(t)}, J^{(t)})$ in terms of the inverse of the previous pivot matrix $\mathbf{A}(I^{(t - 1)}, J^{(t - 1)})$. For simplicity, denote

$$
\begin{aligned}
    \mathbf{P}^{(t - 1)} &= \mathbf{A}(I^{(t - 1)}, J^{(t - 1)}), \\
    \mathbf{Q}^{(t - 1)} &= \mathbf{A}(I^{(t - 1)}, J^{(t - 1)})^{-1}, \\
    \mathbf{P}^{(t)} &= \mathbf{A}(I^{(t)}, J^{(t)}), \\
    \mathbf{Q}^{(t)} &= \mathbf{A}(I^{(t)}, J^{(t)})^{-1}, \\
    \mathbf{c}^{(t)} &= \mathbf{A}(I^{(t - 1)}, j_t), \\
    \mathbf{r}^{(t)} &= \mathbf{A}(i_t, J^{(t - 1)}), \\
    p^{(t)} &= \mathbf{A}(i_t, j_t), \\
\end{aligned}
$$

Then one has

$$
\begin{aligned}
    \mathbf{Q}^{(t)} &= \begin{bmatrix}
        \mathbf{P}^{(t - 1)} & \mathbf{c}^{(t)} \\
        \mathbf{r}^{(t)} & s
    \end{bmatrix}^{-1} \\
    &= \begin{bmatrix}
        \mathbf{Q}^{(t - 1)} + \mathbf{Q}^{(t - 1)} \mathbf{c}^{(t)} s^{-1} \mathbf{r}^{(t)} \mathbf{Q}^{(t - 1)} & -\mathbf{Q}^{(t - 1)} \mathbf{c}^{(t)} s^{-1} \\
        -s^{-1} \mathbf{r}^{(t)} \mathbf{Q}^{(t - 1)} & s^{-1}
    \end{bmatrix}
\end{aligned}
$$

where $s = p^{(t)} - \mathbf{r}^{(t)} \mathbf{Q}^{(t - 1)} \mathbf{c}^{(t)}$.

Further, expand the expression of $\mathbf{\tilde{A}}^{(t)}$ as follows:

$$
\begin{aligned}
    \mathbf{\tilde{A}}^{(t)} &= \mathbf{A}(\mathbb{I}, J^{(t)}) \mathbf{Q}^{(t)} \mathbf{A}(I^{(t)}, \mathbb{J}) \\

    &= \begin{bmatrix}
        \mathbf{A}(\mathbb{I}, J^{(t - 1)}) & \mathbf{A}(\mathbb{I}, j_t)
    \end{bmatrix}

    \begin{bmatrix}
        \mathbf{Q}^{(t - 1)} + \mathbf{Q}^{(t - 1)} \mathbf{c}^{(t)} s^{-1} \mathbf{r}^{(t)} \mathbf{Q}^{(t - 1)} & -\mathbf{Q}^{(t - 1)} \mathbf{c}^{(t)} s^{-1} \\
        -s^{-1} \mathbf{r}^{(t)} \mathbf{Q}^{(t - 1)} & s^{-1}
    \end{bmatrix}

    \begin{bmatrix}
        \mathbf{A}(I^{(t - 1)}, \mathbb{J}) \\
        \mathbf{A}(i_t, \mathbb{J})
    \end{bmatrix} \\

    &= \begin{bmatrix}
        \mathbf{A}(\mathbb{I}, J^{(t - 1)}) \mathbf{Q}^{(t - 1)} + \mathbf{A}(\mathbb{I}, J^{(t - 1)}) \mathbf{Q}^{(t - 1)} \mathbf{c}^{(t)} s^{-1} \mathbf{r}^{(t)} \mathbf{Q}^{(t - 1)} - \mathbf{A}(\mathbb{I}, j_t) s^{-1} \mathbf{r}^{(t)} \mathbf{Q}^{(t - 1)} & -\mathbf{A}(\mathbb{I}, J^{(t - 1)}) \mathbf{Q}^{(t - 1)} \mathbf{c}^{(t)} s^{-1} + \mathbf{A}(\mathbb{I}, j_t) s^{-1}
    \end{bmatrix}

    \begin{bmatrix}
        \mathbf{A}(I^{(t - 1)}, \mathbb{J}) \\
        \mathbf{A}(i_t, \mathbb{J})
    \end{bmatrix} \\

    &= \mathbf{A}(\mathbb{I}, J^{(t - 1)}) \mathbf{Q}^{(t - 1)} \mathbf{A}(I^{(t - 1)}, \mathbb{J}) \\
    &\quad + \mathbf{A}(\mathbb{I}, J^{(t - 1)}) \mathbf{Q}^{(t - 1)} \mathbf{c}^{(t)} s^{-1} \mathbf{r}^{(t)} \mathbf{Q}^{(t - 1)} \mathbf{A}(I^{(t - 1)}, \mathbb{J}) \\
    &\quad - \mathbf{A}(\mathbb{I}, j_t) s^{-1} \mathbf{r}^{(t)} \mathbf{Q}^{(t - 1)} \mathbf{A}(I^{(t - 1)}, \mathbb{J}) \\
    &\quad -\mathbf{A}(\mathbb{I}, J^{(t - 1)}) \mathbf{Q}^{(t - 1)} \mathbf{c}^{(t)} s^{-1} \mathbf{A}(i_t, \mathbb{J}) \\
    &\quad + \mathbf{A}(\mathbb{I}, j_t) s^{-1} \mathbf{A}(i_t, \mathbb{J}) \\

    &= \mathbf{\tilde{A}}^{(t - 1)} \\
    &\quad + s^{-1} \mathbf{A}(\mathbb{I}, J^{(t - 1)}) \mathbf{A}(I^{(t - 1)}, J^{(t - 1)})^{-1} \mathbf{A}(I^{(t - 1)}, j_t) \mathbf{A}(i_t, J^{(t - 1)}) \mathbf{A}(I^{(t - 1)}, J^{(t - 1)})^{-1} \mathbf{A}(I^{(t - 1)}, \mathbb{J}) \\
    &\quad - s^{-1} \mathbf{A}(\mathbb{I}, j_t) \mathbf{A}(i_t, J^{(t - 1)}) \mathbf{A}(I^{(t - 1)}, J^{(t - 1)})^{-1} \mathbf{A}(I^{(t - 1)}, \mathbb{J}) \\
    &\quad - s^{-1} \mathbf{A}(\mathbb{I}, J^{(t - 1)}) \mathbf{A}(I^{(t - 1)}, J^{(t - 1)})^{-1} \mathbf{A}(I^{(t - 1)}, j_t) \mathbf{A}(i_t, \mathbb{J}) \\
    &\quad + s^{-1} \mathbf{A}(\mathbb{I}, j_t) \mathbf{A}(i_t, \mathbb{J}) \\

    &= \mathbf{\tilde{A}}^{(t - 1)} \\
    &\quad + s^{-1} \mathbf{\tilde{A}}^{(t - 1)}(\mathbb{I}, j_t) \mathbf{\tilde{A}}^{(t - 1)}(i_t, \mathbb{J}) \\
    &\quad - s^{-1} \mathbf{A}(\mathbb{I}, j_t) \mathbf{\tilde{A}}^{(t - 1)}(i_t, \mathbb{J}) \\
    &\quad - s^{-1} \mathbf{\tilde{A}}^{(t - 1)}(\mathbb{I}, j_t) \mathbf{A}(i_t, \mathbb{J}) \\
    &\quad + s^{-1} \mathbf{A}(\mathbb{I}, j_t) \mathbf{A}(i_t, \mathbb{J}) \\

    &= \mathbf{\tilde{A}}^{(t - 1)} + s^{-1} (\mathbf{A}(\mathbb{I}, j_t) - \mathbf{\tilde{A}}^{(t - 1)}(\mathbb{I}, j_t)) (\mathbf{A}(i_t, \mathbb{J}) - \mathbf{\tilde{A}}^{(t - 1)}(i_t, \mathbb{J})).
\end{aligned}
$$

Note that

$$
\begin{aligned}
    s = p^{(t)} - \mathbf{r}^{(t)} \mathbf{Q}^{(t - 1)} \mathbf{c}^{(t)} = \mathbf{A}(i_t, j_t) - \mathbf{\tilde{A}}^{(t - 1)}(i_t, j_t) &= \mathcal{E}^{(t - 1)}(i_t, j_t), \\
    \mathbf{A}(\mathbb{I}, j_t) - \mathbf{\tilde{A}}^{(t - 1)}(\mathbb{I}, j_t) &= \mathcal{E}^{(t - 1)}(\mathbb{I}, j_t), \\
    \mathbf{A}(i_t, \mathbb{J}) - \mathbf{\tilde{A}}^{(t - 1)}(i_t, \mathbb{J}) &= \mathcal{E}^{(t - 1)}(i_t, \mathbb{J}),
\end{aligned}
$$

which leads to the final incremental update formula

$$
\mathbf{\tilde{A}}^{(t)} = \mathbf{\tilde{A}}^{(t - 1)} + \frac{1}{\mathcal{E}^{(t - 1)}(i_t, j_t)} \mathcal{E}^{(t - 1)}(\mathbb{I}, j_t) \mathcal{E}^{(t - 1)}(i_t, \mathbb{J}).
$$

Then, the algorithm goes as follows:

1. Initialize $I = J = \emptyset, \mathbf{\tilde{A}}^{(0)} = \mathbf{0}, \mathcal{E}^{(0)} = \mathbf{A}$.

2. At step $t$, find a new pivot $(i_t, j_t)$ that maximizes $\mathcal{E}^{(t - 1)}(i, j)$.

3. Add the point $(i_t, j_t)$ to the sets $I$ and $J$.

4. Update the matrices:
    - $\mathcal{D}^{(t)} = \frac{1}{\mathcal{E}^{(t - 1)}(i_t, j_t)} \mathcal{E}^{(t - 1)}(\mathbb{I}, j_t) \mathcal{E}^{(t - 1)}(i_t, \mathbb{J})$;
    - $\mathbf{\tilde{A}}^{(t)} = \mathbf{\tilde{A}}^{(t - 1)} + \mathcal{D}^{(t)}$;
    - $\mathcal{E}^{(t)} = \mathcal{E}^{(t - 1)} - \mathcal{D}^{(t)}$.

5. Repeat steps 2-4 until $|\mathcal{E}^{(t - 1)}(i_t, j_t)| \leq \epsilon$ or the maximum rank is reached.

## Efficient ACA Algorithm

The ACA algorithm skips the inverse computation of the pivot matrix, but it still requires to store the whole interpolation matrix $\mathbf{\tilde{A}}$ and the error matrix $\mathcal{E}$, which can be memory consuming for large matrices.

To address this issue, one can represent the interpolation matrix $\mathbf{\tilde{A}}$ as a sum of rank-1 matrices, which only requires to store the factors of the rank-1 matrices instead of the whole matrix.

That is, let

$$
\begin{aligned}
    e_p^{(t)} &= \mathcal{E}^{(t - 1)}(i_t, j_t), \\
    \mathbf{e}_c^{(t)} &= \mathcal{E}^{(t - 1)}(\mathbb{I}, j_t), \\
    \mathbf{e}_r^{(t)} &= \mathcal{E}^{(t - 1)}(i_t, \mathbb{J}),
\end{aligned}
$$

then one has

$$
\mathbf{\tilde{A}}^{(t)} = \sum_{k = 1}^{t} \frac{1}{e_p^{(k)}} \mathbf{e}_c^{(k)} \mathbf{e}_r^{(k)}.
$$

Then, the algorithm goes as follows:

1. Initialize $I = J = \emptyset$.

2. At step $t$, find a new pivot $(i_t, j_t)$ that maximizes $\mathcal{E}^{(t - 1)}(i, j)$.
    - For ``full search``, compute $\mathcal{E}^{(t - 1)} = \mathbf{A} - \sum_{k = 1}^{t - 1} \frac{1}{e_p^{(k)}} \mathbf{e}_c^{(k)} \mathbf{e}_r^{(k)}$;
    - For ``rook search``:
        - compute $\mathcal{E}^{(t - 1)}(\mathbb{I}, j_t) = \mathbf{A}(\mathbb{I}, j_t) - \sum_{k = 1}^{t - 1} \frac{1}{e_p^{(k)}} \mathbf{e}_c^{(k)} \mathbf{e}_r^{(k)}(\mathbb{I}, j_t)$,
        - compute $\mathcal{E}^{(t - 1)}(i_t, \mathbb{J}) = \mathbf{A}(i_t, \mathbb{J}) - \sum_{k = 1}^{t - 1} \frac{1}{e_p^{(k)}} \mathbf{e}_c^{(k)}(i_t) \mathbf{e}_r^{(k)}$.

3. Add the point $(i_t, j_t)$ to the sets $I$ and $J$.

4. Repeat steps 2-3 until $|\mathcal{E}^{(t - 1)}(i_t, j_t)| \leq \epsilon$ or the maximum rank is reached.





