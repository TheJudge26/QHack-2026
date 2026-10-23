# Challenge: My First Quantum Neural Network (100 pts)

## 1. Problem Overview
Quantum Neural Networks (QNNs) combine parameterized quantum circuits with classical optimization algorithms to learn patterns and solve machine learning tasks. A variational quantum circuit typically consists of:
1. **Data Encoding Block $S(x)$**: Maps classical inputs $x$ into quantum states (quantum feature map).
2. **Trainable / Variational Block $A(\vec{\theta})$**: Parameterized unitary transformations tuned via optimization.
3. **Measurement / Observable**: Expectation value measurement $\langle \hat{B} \rangle$ producing classical outputs.

In this challenge, we construct a single-qubit, single-layer quantum neural network:
* State initialization: $|0\rangle$
* Encoding: $S(x) = R_X(x)$
* Trainable block: $A(\vec{\theta}) = R_Z(\theta_3) R_Y(\theta_2) R_Z(\theta_1)$
* Measurement: Expectation value of Pauli $Z$ observable, $\langle \hat{\sigma}_z \rangle$.

---

## 2. Derivation & Circuit Architecture

### 1. Quantum State Preparation and Encoding
Starting from the ground state $|0\rangle$, the single-qubit encoding circuit applies an $R_X$ rotation parameterized by the input scalar $x$:
$$|\psi_0(x)\rangle = S(x) |0\rangle = R_X(x) |0\rangle = \cos\left(\frac{x}{2}\right) |0\rangle - i \sin\left(\frac{x}{2}\right) |1\rangle$$

### 2. Parameterized Processing Block
The variational block $A(\vec{\theta})$ applies an arbitrary single-qubit rotation defined by Euler angles $\vec{\theta} = (\theta_1, \theta_2, \theta_3)$:
$$|\psi(x, \vec{\theta})\rangle = A(\vec{\theta}) |\psi_0(x)\rangle = R_Z(\theta_3) R_Y(\theta_2) R_Z(\theta_1) |\psi_0(x)\rangle$$

* $R_Z(\theta_1)$ on wire 0
* $R_Y(\theta_2)$ on wire 0
* $R_Z(\theta_3)$ on wire 0

### 3. Observable Measurement
The circuit outputs the expectation value of the Pauli-$Z$ observable:
$$\langle \hat{\sigma}_z \rangle = \langle \psi(x, \vec{\theta}) | \sigma_z | \psi(x, \vec{\theta}) \rangle$$

---

## 3. Implementation

```python
import pennylane as qp
import pennylane.numpy as np

dev = qp.device("default.qubit", wires=1)

def training_circuit(theta_vec):
    qp.RZ(theta_vec[0], wires=0)
    qp.RY(theta_vec[1], wires=0)
    qp.RZ(theta_vec[2], wires=0)

def encoding_circuit(x):
    qp.RX(x, wires=0)

@qp.qnode(dev)
def quantum_neural_network(x, theta_vec):
    encoding_circuit(x)
    training_circuit(theta_vec)
    return qp.expval(qp.PauliZ(0))
```
