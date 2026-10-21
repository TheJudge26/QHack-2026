# Challenge: Playing Period (100 pts)

## 1. Problem Overview
The Quantum Fourier Transform (QFT) maps computational basis states $|x\rangle$ to the Fourier basis:
$$|x\rangle \longrightarrow \frac{1}{\sqrt{2^n}} \sum_{y=0}^{2^n - 1} e^{2\pi i x y / 2^n} |y\rangle$$

In this challenge, we construct the **4-qubit QFT** using only elementary gates:
* `qp.Hadamard`
* `qp.ControlledPhaseShift`
* `qp.SWAP`

Using the built-in `qp.QFT` template is disallowed.

---

## 2. Derivation & Circuit Architecture

For an $n$-qubit system with binary representation $x = x_0 2^{n-1} + \dots + x_{n-1} 2^0 = [x_0 x_1 \dots x_{n-1}]$, the QFT can be factored into a product state:
$$\text{QFT}|x_0 x_1 \dots x_{n-1}\rangle = \frac{1}{\sqrt{2^n}} \bigotimes_{j=0}^{n-1} \left( |0\rangle + e^{2\pi i (0.x_j x_{j+1} \dots x_{n-1})} |1\rangle \right)$$

where $0.x_j x_{j+1} \dots x_{n-1} = \sum_{k=j}^{n-1} x_k 2^{-(k - j + 1)}$ represents binary fractions.

### Gate Sequence:
For each qubit `target` from $0$ to $n - 1$:
1. Apply **Hadamard** on `target`: creates the superposition $\frac{|0\rangle + (-1)^{x_{\text{target}}} |1\rangle}{\sqrt{2}} = \frac{|0\rangle + e^{2\pi i (0.x_{\text{target}})} |1\rangle}{\sqrt{2}}$.
2. Apply **controlled phase shifts** from higher index qubits `control = target + 1, ..., n - 1`:
   $$\phi = \frac{2\pi}{2^{(\text{control} - \text{target} + 1)}} = \frac{\pi}{2^{\text{control} - \text{target}}}$$
   When both qubits are $|1\rangle$, this applies a phase of $e^{i\phi}$.

### Qubit Reversal via SWAP:
The standard QFT circuit outputs the qubits in **bit-reversed order**:
* Qubit $0$ ends up with the phase depending on $0.x_0 x_1 x_2 x_3$ (most significant frequency component).
* Qubit $3$ ends up with the phase depending only on $0.x_3$.

To match the standard convention and PennyLane's `qp.QFT`:
* Swap qubit $0$ and qubit $3$: `qp.SWAP(wires=[0, 3])`
* Swap qubit $1$ and qubit $2$: `qp.SWAP(wires=[1, 2])`

---

## 3. Implementation
```python
def qft_four_qubits():
    for target in range(4):
        qp.Hadamard(wires=target)
        for control in range(target + 1, 4):
            phi = np.pi / (2 ** (control - target))
            qp.ControlledPhaseShift(phi, wires=[control, target])

    qp.SWAP(wires=[0, 3])
    qp.SWAP(wires=[1, 2])
```
