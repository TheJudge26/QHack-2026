# The Scouting Report (100 pts)

## Problem Overview
On quantum hardware, wavefunction amplitudes cannot be read out directly. The state of a quantum system is reconstructed through **Quantum State Tomography (QST)** by measuring the expectation values of an informationally complete set of observables.

For a single qubit (or the reduced state of one qubit in an $n$-qubit register), any density matrix $\rho$ can be uniquely parameterized in the Pauli basis:
$$\rho = \frac{1}{2}\left(I + r_x X + r_y Y + r_z Z\right)$$

where $\vec{r} = (r_x, r_y, r_z) \in \mathbb{R}^3$ is the **Bloch vector**.

---

## 1. Bloch Vector Components
Using the orthogonality relation of Pauli matrices $\text{Tr}(\sigma_i \sigma_j) = 2 \delta_{ij}$, the Bloch components are obtained directly from single-qubit expectation values:
$$r_i = \text{Tr}(\rho \sigma_i) = \langle \sigma_i \rangle, \quad \sigma_i \in \{X, Y, Z\}$$

For a target wire in an $n$-qubit system prepared by a black-box circuit `prep()`:
- $r_x = \langle X_{\text{target}} \rangle$
- $r_y = \langle Y_{\text{target}} \rangle$
- $r_z = \langle Z_{\text{target}} \rangle$

Each expectation value is measured using a dedicated QNode with `qp.expval(qp.PauliX(target_wire))`, `qp.expval(qp.PauliY(target_wire))`, and `qp.expval(qp.PauliZ(target_wire))`.

---

## 2. Quantum State Purity
The purity of a state $\rho$ measures its mixedness / entanglement with the unmeasured environment/subsystem:
$$\gamma = \text{Tr}(\rho^2) = \frac{1}{2}\left(1 + \|\vec{r}\|^2\right) = \frac{1}{2}\left(1 + r_x^2 + r_y^2 + r_z^2\right)$$

- **Pure state** ($\|\vec{r}\| = 1$): $\text{Tr}(\rho^2) = 1$. The target qubit is in a product state with the rest of the register.
- **Maximally mixed state** ($\|\vec{r}\| = 0 \implies \rho = \frac{I}{2}$): $\text{Tr}(\rho^2) = \frac{1}{2}$. The target qubit is maximally entangled with the remaining qubits (e.g., in a Bell state).
- **Partially mixed state** ($0 < \|\vec{r}\| < 1$): $\frac{1}{2} < \text{Tr}(\rho^2) < 1$.

---

## 3. Implementation
```python
def bloch_and_purity(prep, num_wires, target_wire):
    dev = qp.device("default.qubit", wires=num_wires)

    @qp.qnode(dev)
    def circuit_x():
        prep()
        return qp.expval(qp.PauliX(target_wire))

    @qp.qnode(dev)
    def circuit_y():
        prep()
        return qp.expval(qp.PauliY(target_wire))

    @qp.qnode(dev)
    def circuit_z():
        prep()
        return qp.expval(qp.PauliZ(target_wire))

    r_x = float(circuit_x())
    r_y = float(circuit_y())
    r_z = float(circuit_z())
    r = [r_x, r_y, r_z]

    purity = float(0.5 * (1.0 + r_x**2 + r_y**2 + r_z**2))

    return r, purity
```
