# Challenge: Implementing POVMs (400 pts)

## 1. Problem Overview
In standard quantum mechanics, projective measurements (von Neumann measurements) are described by a set of orthogonal projection operators $\{P_i\}$ satisfying $P_i P_j = \delta_{ij} P_i$ and $\sum_i P_i = I$. The probability of obtaining outcome $i$ on state $|\psi\rangle$ is $p(i) = \langle\psi| P_i |\psi\rangle$.

In general quantum measurement theory, any physically realizable measurement can be described by a **Positive Operator-Valued Measure (POVM)**: a collection of positive semi-definite Hermitian operators $\{E_i\}$ satisfying:
$$\sum_{i=0}^N E_i = I, \qquad E_i \ge 0, \qquad p(i) = \langle\psi| E_i |\psi\rangle$$
Unlike projective measurements, the elements $E_i$ do not need to be mutually orthogonal projectors, allowing for more outcomes than the Hilbert space dimension.

In this challenge, we implement a 3-outcome POVM on a single-qubit system $S$ (wire 1):
$$E_0 = \alpha |1\rangle\langle 1| = \begin{pmatrix} 0 & 0 \\ 0 & \alpha \end{pmatrix}$$
$$E_1 = \beta |-\rangle\langle -| = \frac{\beta}{2}\begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix}$$
$$E_2 = I - E_0 - E_1 = \begin{pmatrix} 1 - \frac{\beta}{2} & \frac{\beta}{2} \\ \frac{\beta}{2} & 1 - \alpha - \frac{\beta}{2} \end{pmatrix}$$
where $\alpha, \beta \in [0, 1]$.

By **Naimark's Dilation Theorem**, any POVM can be realized as a projective measurement on a dilated Hilbert space. Here, we couple the system qubit $S$ (wire 1) to an auxiliary ancilla qubit $A$ (wire 0) initialized in $|0\rangle$, apply a two-qubit unitary $U$, and measure in the computational basis $\{|ij\rangle\}_{i,j \in \{0, 1\}}$, such that:
$$p_{00} = \langle\psi| E_0 |\psi\rangle$$
$$p_{01} = \langle\psi| E_1 |\psi\rangle$$
$$p_{10} + p_{11} = \langle\psi| E_2 |\psi\rangle$$

---

## 2. Mathematical Derivation & Dilation Architecture

### 1. Joint State Representation and Isometry $V$
The initial state of the joint 2-qubit system is:
$$|\Psi_{\text{in}}\rangle = |0\rangle_0 \otimes |\psi\rangle_1 = \psi_0 |00\rangle + \psi_1 |01\rangle = \begin{pmatrix} \psi_0 \\ \psi_1 \\ 0 \\ 0 \end{pmatrix}$$

Applying a $4 \times 4$ unitary $U$ to $|\Psi_{\text{in}}\rangle$ gives:
$$|\Psi_{\text{out}}\rangle = U |\Psi_{\text{in}}\rangle = \psi_0 U|00\rangle + \psi_1 U|01\rangle = V |\psi\rangle$$
where $V$ is the $4 \times 2$ isometry formed by the first two columns of $U$:
$$V = \begin{pmatrix} \langle 00|U|00\rangle & \langle 00|U|01\rangle \\ \langle 01|U|00\rangle & \langle 01|U|01\rangle \\ \langle 10|U|00\rangle & \langle 10|U|01\rangle \\ \langle 11|U|00\rangle & \langle 11|U|01\rangle \end{pmatrix} = \begin{pmatrix} M_0 \\ M_1 \\ M_2 \end{pmatrix}$$

### 2. Constraints from POVM Elements
The probability of measuring each basis state is given by the squared magnitude of the projection:

1. **Outcome $|00\rangle$**:
   $$p_{00} = |\langle 00 | U |0\rangle |\psi\rangle|^2 = |V_{00} \psi_0 + V_{01} \psi_1|^2$$
   We require $p_{00} = \langle\psi| E_0 |\psi\rangle = \alpha |\psi_1|^2$ for all states $|\psi\rangle$. This uniquely determines:
   $$V_{00} = 0, \qquad V_{01} = \sqrt{\alpha}$$

2. **Outcome $|01\rangle$**:
   $$p_{01} = |\langle 01 | U |0\rangle |\psi\rangle|^2 = |V_{10} \psi_0 + V_{11} \psi_1|^2$$
   We require $p_{01} = \langle\psi| E_1 |\psi\rangle = \beta |\langle - | \psi\rangle|^2 = \frac{\beta}{2} |\psi_0 - \psi_1|^2$ for all states $|\psi\rangle$. This implies:
   $$V_{10} = \sqrt{\frac{\beta}{2}}, \qquad V_{11} = -\sqrt{\frac{\beta}{2}}$$

3. **Outcome $|10\rangle + |11\rangle$**:
   Let $M_2 = \begin{pmatrix} V_{20} & V_{21} \\ V_{30} & V_{31} \end{pmatrix}$. The sum of probabilities for wire 0 measured in $|1\rangle$ is:
   $$p_{10} + p_{11} = \| M_2 |\psi\rangle \|^2 = \langle\psi| M_2^\dagger M_2 |\psi\rangle$$
   We require $p_{10} + p_{11} = \langle\psi| E_2 |\psi\rangle$ for all $|\psi\rangle$. Thus:
   $$M_2^\dagger M_2 = E_2 = I - E_0 - E_1 = \begin{pmatrix} 1 - \frac{\beta}{2} & \frac{\beta}{2} \\ \frac{\beta}{2} & 1 - \alpha - \frac{\beta}{2} \end{pmatrix}$$

### 3. Construction of $M_2$ via Spectral Decomposition
Because $E_2$ is real symmetric and positive semi-definite ($E_2 \ge 0$), its principal matrix square root $M_2 = \sqrt{E_2}$ is also real symmetric and satisfies $M_2^\dagger M_2 = M_2^2 = E_2$:
$$E_2 = P \Lambda P^T \implies M_2 = P \sqrt{\Lambda} P^T$$
where $\Lambda = \operatorname{diag}(\lambda_1, \lambda_2)$ contains the non-negative eigenvalues of $E_2$ and $P$ is the orthogonal matrix of eigenvectors.

### 4. Orthonormality and Unitary Completion
Notice that:
$$V^\dagger V = M_0^\dagger M_0 + M_1^\dagger M_1 + M_2^\dagger M_2 = E_0 + E_1 + E_2 = I_2$$
Thus, the two column vectors of $V$ are strictly orthonormal in $\mathbb{C}^4$.

To complete $V$ into a full $4 \times 4$ unitary matrix $U = [V \quad W]$, we find an orthonormal basis $W \in \mathbb{C}^{4 \times 2}$ for the orthogonal complement (null space) $\ker(V^\dagger)$:
$$\text{SVD}(V^T) = U_S \Sigma V_h \implies W = (V_h[2:])^T$$
Constructing $U = \begin{pmatrix} V & W \end{pmatrix}$ guarantees:
$$U^\dagger U = I_4$$
Since the initial state has support only on the first two computational basis states, the choice of completion $W$ has zero overlap with $|\Psi_{\text{in}}\rangle$, preserving exact probabilities for all possible input states.

---

## 3. Implementation

```python
import json
import pennylane as qp
import pennylane.numpy as np


def get_povm_unitary(alpha, beta):
    """Constructs a 4x4 unitary matrix implementing Naimark dilation for the POVM."""
    # E_2 = I - E_0 - E_1
    E2 = np.array([
        [1.0 - beta / 2.0, beta / 2.0],
        [beta / 2.0, 1.0 - alpha - beta / 2.0]
    ], dtype=float)

    # Compute square root M_2 = sqrt(E_2) via spectral decomposition
    eigvals, eigvecs = np.linalg.eigh(E2)
    eigvals = np.clip(eigvals, 0.0, None)
    M2 = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    # First two columns of U form an isometry V
    V = np.zeros((4, 2), dtype=float)
    V[0, :] = [0.0, np.sqrt(alpha)]
    V[1, :] = [np.sqrt(beta / 2.0), -np.sqrt(beta / 2.0)]
    V[2:4, :] = M2

    # Orthonormal completion of V to a 4x4 unitary using SVD of V^T
    _, _, Vh = np.linalg.svd(V.T)
    W = Vh[2:].T
    U = np.hstack([V, W])
    return U


def implement_POVM(alpha, beta):
    """Quantum function that implements the protocol for the POVM."""
    U = get_povm_unitary(alpha, beta)
    qp.QubitUnitary(U, wires=[0, 1])
```

---

## 4. Verification & Results

Running the local verification suite:
* **Test Case 0** (`alpha = 0.25, beta = 0.25`): **Correct!**
* **Test Case 1** (`alpha = 0.2, beta = 0.4`): **Correct!**
* **Random State Vectors**: 10 random pure states verified per test case with relative error tolerance $< 10^{-4}$.
