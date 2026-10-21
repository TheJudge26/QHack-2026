# Challenge: Building a Quantum Kernel (100 pts)

## 1. Problem Overview
Kernel methods map classical data points $x$ into a high-dimensional Hilbert space:
$$x \longrightarrow |\phi(x)\rangle = S(x)|0\rangle$$

The quantum kernel represents the transition amplitude (fidelity / squared overlap) between two embedded states:
$$k(x_1, x_2) = |\langle \phi(x_2) | \phi(x_1) \rangle|^2 = |\langle 0 | S(x_2)^\dagger S(x_1) | 0 \rangle|^2$$

Given two sets of datapoints $X_1$ and $X_2$, the task is to construct the Gram / kernel matrix:
$$K_{ij} = k(x_1^{(i)}, x_2^{(j)})$$

---

## 2. Quantum Circuit Architecture

### State Embedding
We encode 4 features into 4 qubits using **Angle Embedding** ($R_X$ rotations):
$$S(x) = \bigotimes_{j=0}^3 R_X(x_j)$$

### Overlap Measurement
To measure $|\langle 0 | S(x_2)^\dagger S(x_1) | 0 \rangle|^2$:
1. Apply $S(x_1)$ to state $|0000\rangle$: `qp.AngleEmbedding(x1, wires=range(4))`
2. Apply $S(x_2)^\dagger$: `qp.adjoint(qp.AngleEmbedding)(x2, wires=range(4))`
3. Measure the expectation value of the projector onto $|0000\rangle$:
   $$\langle 0000 | \psi \rangle \langle \psi | 0000 \rangle = |\langle 0000 | \psi \rangle|^2 = k(x_1, x_2)$$
   using `qp.expval(qp.Projector([0, 0, 0, 0], wires=range(4)))`.

---

## 3. Implementation
```python
@qp.qnode(dev)
def quantum_kernel(x1, x2):
    qp.AngleEmbedding(x1, wires=range(4))
    qp.adjoint(qp.AngleEmbedding)(x2, wires=range(4))
    return qp.expval(qp.Projector([0, 0, 0, 0], wires=range(4)))

def kernel_matrix(X1, X2):
    return np.array([[quantum_kernel(x1, x2) for x2 in X2] for x1 in X1])
```
