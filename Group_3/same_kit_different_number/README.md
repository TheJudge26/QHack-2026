# Challenge: Same Kit, Different Number (100 pts)

## 1. Problem Overview

In quantum mechanics, two unitary operators $U$ and $V$ represent the same physical quantum gate if they differ only by an unobservable global phase factor:
$$V = e^{i\alpha} U \quad (\alpha \in \mathbb{R})$$

For any state $|\psi\rangle$ and any observable $O$, measurement probabilities depend strictly on squared amplitudes:
$$|\langle\phi| V |\psi\rangle|^2 = |e^{i\alpha} \langle\phi| U |\psi\rangle|^2 = |e^{i\alpha}|^2 |\langle\phi| U |\psi\rangle|^2 = |\langle\phi| U |\psi\rangle|^2$$
Thus, no single-qubit measurement can distinguish $V$ from $U$. For example, $H$ and $-H$ represent the exact same gate ($e^{i\pi} H = -H$).

### Converting Global Phase into an Observable Relative Phase
While a global phase on a state drops out of measurements, controlling the gate relative to an ancilla wire converts the global phase into an observable relative phase between superposition branches (the foundational mechanism of **phase kickback** and quantum eigenvalue estimation).

Given:
$$M = V U^\dagger$$
If $V = e^{i\alpha} U$, then $M = e^{i\alpha} I$.

Consider a 2-qubit circuit with:
- **Control wire (wire 0):** Prepared in $|+\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}$ via a Hadamard gate.
- **Target wire (wire 1):** Prepared in $|0\rangle$.

Applying controlled-$M$ activates $M$ only when the control wire is $|1\rangle$:
$$\operatorname{ctrl}(M) \left( \frac{|0\rangle + |1\rangle}{\sqrt{2}} \otimes |0\rangle \right) = \frac{|0\rangle \otimes |0\rangle + |1\rangle \otimes (M |0\rangle)}{\sqrt{2}}$$

When $M = e^{i\alpha} I$:
$$= \frac{|0\rangle + e^{i\alpha} |1\rangle}{\sqrt{2}} \otimes |0\rangle$$
The target remains in $|0\rangle$, while the control qubit acquires the relative phase $e^{i\alpha}$.

Applying a second Hadamard gate to the control qubit maps this relative phase into interference populations:
$$H \left( \frac{|0\rangle + e^{i\alpha} |1\rangle}{\sqrt{2}} \right) = \cos\left(\frac{\alpha}{2}\right) e^{i\alpha/2} |0\rangle - i \sin\left(\frac{\alpha}{2}\right) e^{i\alpha/2} |1\rangle$$

The probability of measuring outcome $1$ on the control qubit is:
$$P(1) = \sin^2\left(\frac{\alpha}{2}\right)$$

---

## 2. Mathematical Derivation & Algorithmic Steps

1. **Matrix Product $M$:**
   Compute:
   $$M = V U^\dagger = V U^{*T}$$
2. **Phase Extraction & Gate Equivalence:**
   Extract candidate phase from the leading diagonal entry:
   $$\alpha_{\text{cand}} = \operatorname{Arg}(M_{0,0}) \pmod{2\pi} \in [0, 2\pi)$$
   Check if $M$ is proportional to the identity:
   $$\text{same\_gate} = \left( M \approx e^{i \alpha_{\text{cand}}} I_2 \right)$$
   If `same_gate` is `True`, $\alpha = \alpha_{\text{cand}}$; otherwise, $\alpha = 0.0$.
3. **Phase Kickback Circuit Simulation:**
   Construct and execute a 2-qubit PennyLane QNode on `default.qubit`:
   - Initialize wires $0$ and $1$ in $|00\rangle$.
   - Apply `qp.Hadamard(wires=0)`.
   - Apply `qp.ctrl(qp.QubitUnitary(M, wires=1), control=0)`.
   - Apply `qp.Hadamard(wires=0)`.
   - Return `qp.probs(wires=0)`.
   The measured probability $P(1)$ is `probs[1]`.

---

## 3. Implementation

Located in [`same_kit_different_number.py`](file:///d:/QHack-2026/Group_3/same_kit_different_number/same_kit_different_number.py):

```python
import json
import pennylane as qp
import pennylane.numpy as np


def same_gate_and_phase(u_matrix, v_matrix):
    """Decides whether two unitaries are the same gate, and exposes the phase."""
    dev = qp.device("default.qubit", wires=2)

    # Step 1: form M = V U^dagger.
    M = v_matrix @ u_matrix.conj().T

    # Step 2: read alpha off leading entry, fold into [0, 2*pi)
    alpha_cand = float(np.angle(M[0, 0]) % (2 * np.pi))
    same_gate = bool(np.allclose(M, np.exp(1j * alpha_cand) * np.eye(2), atol=1e-5))
    alpha = alpha_cand if same_gate else 0.0

    # Step 3: build the phase kickback circuit.
    @qp.qnode(dev)
    def circuit():
        qp.Hadamard(wires=0)
        qp.ctrl(qp.QubitUnitary(M, wires=1), control=0)
        qp.Hadamard(wires=0)
        return qp.probs(wires=0)

    probs = circuit()
    p_one = float(probs[1])

    return same_gate, alpha, p_one
```

---

## 4. Verification & Validation

Running the automated test suite locally:

```bash
python Group_3/same_kit_different_number/same_kit_different_number.py
```

### Public Test Cases
1. **Case 1: $U = H, V = -H$**
   - $M = -H H^\dagger = -I = e^{i\pi} I$.
   - `same_gate` = `True`.
   - $\alpha = \pi \approx 3.14159265359$.
   - $P(1) = \sin^2(\pi/2) = 1.0$.
   - **Output:** `[true, 3.141592653589793, 1.0]` $\implies$ Correct!

2. **Case 2: $U = X, V = Z$**
   - $M = Z X = -i Y \ne e^{i\alpha} I$.
   - `same_gate` = `False`.
   - $\alpha = 0.0$.
   - Controlled $-i Y$ creates a Bell state $\frac{|00\rangle - |11\rangle}{\sqrt{2}}$, which after Hadamard on wire 0 gives $P(1) = 0.5$.
   - **Output:** `[false, 0.0, 0.5]` $\implies$ Correct!
