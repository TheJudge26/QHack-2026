# Keep your Own Score (100 pts)

## Problem Overview
Quantum hardware inherently performs measurements in the computational ($Z$) basis ($|0\rangle \to 0, |1\rangle \to 1$). When measuring arbitrary Pauli strings $P = \bigotimes_{i=0}^{n-1} P_i$ where $P_i \in \{I, X, Y, Z\}$, expectation values are never computed by "measuring" an abstract operator directly; they are calculated via post-processing classical bitstrings obtained from single-qubit basis changes.

In this challenge, we measure Pauli words on a 3-qubit state without using `qp.expval`, relying solely on computational basis probabilities `qp.probs(wires=[0, 1, 2])` and bitwise parity arithmetic.

---

## 1. Single-Qubit Basis Rotations
Any single-qubit Pauli observable $P$ can be measured in the computational basis by finding a unitary $U$ that rotates the eigenvectors of $P$ to the eigenvectors of $Z$:
$$U^\dagger Z U = P \iff Z = U P U^\dagger$$

- **For $Z$ and $I$**:
  $$Z = Z \implies U = I \quad (\text{"none"})$$
  No gate needed.
- **For $X$**:
  $$H Z H = X \implies U = H \quad (\text{"H"})$$
  Applying a Hadamard gate before measurement.
- **For $Y$**:
  $$S X S^\dagger = Y \implies (H S^\dagger)^\dagger Z (H S^\dagger) = S (H Z H) S^\dagger = S X S^\dagger = Y$$
  Thus, $U = H S^\dagger$ ($\text{"SH"}$): apply $S^\dagger$ (`qp.adjoint(qp.S(w))`) followed by Hadamard (`qp.Hadamard(w)`).

---

## 2. Multi-Qubit Parity Arithmetic
After applying the basis changes, measuring in the computational basis yields a 3-bit outcome $b = (b_0 b_1 b_2)_2 \in \{0, \dots, 7\}$.
The eigenvalue of $Z_w$ corresponding to bit $b_w \in \{0, 1\}$ is $(-1)^{b_w}$:
$$Z_w |b_w\rangle = (-1)^{b_w} |b_w\rangle$$

The eigenvalue of the tensor product $\bigotimes_{w} P_w$ on basis state $|b\rangle$ is the product of eigenvalues for non-identity wires:
$$s_b = \prod_{w: P_w \neq I} (-1)^{b_w} = (-1)^{\sum_{w: P_w \neq I} b_w}$$

- If $P_w = I$, wire $w$ does **not** contribute to the sign (its parity is ignored).
- If $P_w = Z$, wire $w$ **does** contribute to the sign even though no basis rotation was applied.

---

## 3. Expectation Value
The expectation value is simply the parity-weighted average of the outcome probabilities:
$$\langle P \rangle = \sum_{b=0}^{7} s_b \cdot p_b$$

---

## Code Implementation
```python
def measure_pauli_word(word, angles):
    letter_to_basis = {"I": "none", "Z": "none", "X": "H", "Y": "SH"}
    basis = [letter_to_basis[letter] for letter in word]

    dev = qp.device("default.qubit", wires=3)

    @qp.qnode(dev)
    def circuit():
        prepare(angles)
        for wire, change in enumerate(basis):
            if change == "H":
                qp.Hadamard(wires=wire)
            elif change == "SH":
                qp.adjoint(qp.S(wires=wire))
                qp.Hadamard(wires=wire)
        return qp.probs(wires=[0, 1, 2])

    probabilities = circuit()

    carried = [wire for wire, letter in enumerate(word) if letter != "I"]
    signs = [
        int((-1) ** sum((b >> (2 - wire)) & 1 for wire in carried))
        for b in range(8)
    ]

    value = float(sum(p * s for p, s in zip(probabilities, signs)))
    return basis, signs, value
```
