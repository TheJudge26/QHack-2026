# Challenge: Fermionic VQE (200 pts)

## 1. Problem Overview
The Variational Quantum Eigensolver (VQE) prepares candidate ground states to find the minimum eigenvalue of a Hamiltonian.

In this challenge, we construct a fermionic Hamiltonian:
$$H = c_0 a_0 a_0^\dagger + c_1 a_1 a_1^\dagger + c_2 a_2 a_2^\dagger$$
where $a_i$ is the fermionic annihilation operator and $a_i^\dagger$ is the fermionic creation operator on orbital/mode $i$.

We then:
1. Map $H$ to a qubit Hamiltonian using the **Jordan-Wigner transformation** (`qp.jordan_wigner` / `qp.fermi.jordan_wigner`).
2. Construct a parameterized VQE ansatz on 6 qubits.
3. Optimize the circuit parameters using a gradient descent optimizer to discover the minimum eigenvalue.

---

## 2. Quantum Physics & Mathematical Derivation

### Fermionic Operators to Qubits
The anti-commutation relations for fermions are:
$$\{a_i, a_j^\dagger\} = \delta_{ij}$$
For the same mode $i$:
$$a_i a_i^\dagger = 1 - a_i^\dagger a_i = 1 - n_i$$
where $n_i = a_i^\dagger a_i$ is the number operator (eigenvalues $0$ or $1$).

Under the **Jordan-Wigner mapping**:
$$a_i = \left(\bigotimes_{k=0}^{i-1} Z_k\right) \otimes \frac{X_i + i Y_i}{2}$$
$$a_i^\dagger = \left(\bigotimes_{k=0}^{i-1} Z_k\right) \otimes \frac{X_i - i Y_i}{2}$$

Multiplying them gives:
$$a_i a_i^\dagger = \frac{1}{2} (I + Z_i)$$

Notice:
* If qubit $i$ is in state $|0\rangle$, $\langle Z_i \rangle = +1$, so $\langle a_i a_i^\dagger \rangle = 1$.
* If qubit $i$ is in state $|1\rangle$, $\langle Z_i \rangle = -1$, so $\langle a_i a_i^\dagger \rangle = 0$.

### Handling Complex Coefficients
The Jordan-Wigner implementation in PennyLane returns coefficients as complex numbers (e.g. `(0.5 + 0j)`). To construct a valid Hermitian Hamiltonian, we cast the coefficients of the resulting `PauliSentence` to real floats:
```python
H_ps_real = qp.pauli.PauliSentence({k: float(v.real) for k, v in H_ps.items()})
```

### VQE Ansatz & Optimization
Since the Hamiltonian is diagonal in the computational basis, independent single-qubit rotations `qp.RY(params[i], wires=i)` can rotate each qubit between $|0\rangle$ and $|1\rangle$:
* For $c_i < 0$, the optimizer rotates qubit $i$ towards $|0\rangle$ (energy contribution $c_i$).
* For $c_i > 0$, the optimizer rotates qubit $i$ towards $|1\rangle$ (energy contribution $0$).

Using `qp.GradientDescentOptimizer(stepsize=0.3)` converges stably to the exact ground state energy in <80 iterations.
