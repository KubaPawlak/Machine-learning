# Collaborative filtering

## Model

We changed some of the notation from the lecture example, to try make it more intuitive what different variables mean.
This section serves to precisely formulate the meanings of values used in the later sections.

Users will be denoted by $u$, where $U$ is the total number of users.
Movies will be denoted by $m$, where $M$ is the total number of movies.

### Movie ratings

Let $y_{m,u}$, or alternatively $y[m,u]$ denote the rating of user $m$ by user $u$.

Therefore, $y$ will be an $M \times U$ matrix:
$$ y =
\begin{bmatrix}
y_{1,1} & y_{1,2} & \cdots & y_{1,U} \\
y_{2,1} & y_{2,2} & \cdots & y_{2,U} \\
\vdots & \vdots & \ddots & \vdots \\
y_{M,1} & y_{M,2} & \cdots & y_{M,U}
\end{bmatrix}
$$

Valid ratings are integers from 0 to 5.
If $y_{m,u}$ takes value of NaN, it will denote that user $u$ did not rate movie $m$.

### Movie features

Let $N$ be the total number of features.
Then, $x_{m,n}$ will be the $n$-th feature of the movie $m$.

As such, $p$ will be an $M \times N$ matrix:
$$ x =
\begin{bmatrix}
x_{1,1} & x_{1,2} & \cdots & x_{1,N} \\
x_{2,1} & x_{2,2} & \cdots & x_{2,N} \\
\vdots & \vdots & \ddots & \vdots \\
x_{M,1} & x_{M,2} & \cdots & x_{M,N}
\end{bmatrix}
$$

### User parameters

Each user $u$ will have a set of parameters $p_u$ corresponding to the movie features,
plus one additional feature $p_{u,0}$

$$ p =
\begin{bmatrix}
p_{1,0} & p_{1,1} & p_{1,2} & \cdots & p_{1,N} \\
p_{2,0} & p_{2,1} & p_{2,2} & \cdots & p_{2,N} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
p_{U,0} & p_{U,1} & p_{U,2} & \cdots & p_{U,N}
\end{bmatrix}
$$

## Calculating predictions

Predicted rating of movie $m$ by user $u$ will be denoted as $\hat{y}(m,u)$
It will be calculated in the following way:
$$
\hat{y}(m,u) = p_{u,0} + \sum\limits_{n=1}^N p_{u,n} \cdot x_{m,n}
$$

Alternatively, as a dot product:
$$
\hat{y}(m,u) = p_{u,0} + p[u,1\!:] \cdot x[m,:],
$$
where
$$
p[u,1\!:] = \begin{bmatrix}
p_{u,1} & p_{u,2} & \cdots & p_{u,N}
\end{bmatrix},$$
$$
x[m,:] = \begin{bmatrix}
x_{1,1} & x_{1,2} & \cdots & x_{1,N}
\end{bmatrix}.
$$
In code:

```python
def prediction(m, u):
    return p[u, 0] + np.dot(p[u, 1:], x[m, :])
```

## Calculating errors

The error function will be as follows:
$$
Q(p,x) = \frac{1}{2} \sum\limits_{m,u : y[m,u] \neq -1} (\hat{y}(m,u) - y[m,u])^2
$$

## Partial derivatives

### Zeroth user parameter

$$
\frac{\partial Q}{\partial p_{u, 0}} = \sum_{m : y[m,u] \neq -1} \big(\hat{y}[m, u] - y[m, u]\big)
$$

```python
for m, u in existing_ratings():
    error = error(u, m)
    grad_p[u, 0] += error
```

### Other user parameters ($1, \ldots, N$)

$$
\frac{\partial Q}{\partial p_{u, n}} = \sum_{m : y[m,u] \neq -1} \big(\hat{y}[m, u] - y[m, u]\big) \cdot x[m, n]
$$

```python
for m, u in existing_ratings():
    error = error(u, m)
    grad_p[u, 1:] += error * x[m, :]
```

### Movie parameters

$$
\frac{\partial Q}{\partial x_{m, n}} = \sum_{u : y[m,u] \neq -1} \big(\hat{y}[m, u] - y[m, u]\big) \cdot p[u, n]
$$

```python
for m, u in existing_ratings():
    error = error(u, m)
    grad_x[m, :] += error * p[u, 1:] 
```

