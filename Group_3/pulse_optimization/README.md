# Challenge: Pulse Optimization (400 pts)

## 1. Problem Overview
Pulse programming allows direct time-dependent electromagnetic control of quantum degrees of freedom at the physical hardware layer. The dynamics of the quantum state $|\psi(t)\rangle$ are governed by the time-dependent Schrödinger equation:
$$\frac{\partial}{\partial t} |\psi(t)\rangle = -i H(t) |\psi(t)\rangle$$

In this challenge, we consider a single-qubit system governed by a time-dependent Hamiltonian parameterized by $\vec{p} = [p_0, p_1, p_2]$:
$$H(t) = p_0 t X + p_1 t^2 Y + p_2 t^3 Z$$
where $t = 2$ is the total evolution time and $X, Y, Z$ are the Pauli matrices on wire 0.

Starting from the initial state $|0\rangle$, our goal is to optimize the parameters $\vec{p}$ to prepare a given target state $|\psi\rangle$ by maximizing the quantum fidelity (or minimizing the negative fidelity):
$$F = |\langle \psi | \mathcal{T}\exp\left(-i \int_0^2 H(t) dt\right) |0\rangle|^2$$

---

## 2. Theoretical Background & Implementation

### 1. Parametrized Hamiltonian in PennyLane
PennyLane's pulse module (`qp.pulse`) represents time-dependent Hamiltonians as linear combinations of static operators multiplied by scalar time-dependent functions $f_i(p_i, t)$:
* $f_0(p_0, t) = p_0 \cdot t$ multiplying $\hat{\sigma}_x$
* $f_1(p_1, t) = p_1 \cdot t^2$ multiplying $\hat{\sigma}_y$
* $f_2(p_2, t) = p_2 \cdot t^3$ multiplying $\hat{\sigma}_z$

```python
def hamiltonian():
    def f0(params, t):
        return params * t

    def f1(params, t):
        return params * (t ** 2)

    def f2(params, t):
        return params * (t ** 3)

    return f0 * qp.PauliX(0) + f1 * qp.PauliY(0) + f2 * qp.PauliZ(0)
```

### 2. Time Evolution QNode
The circuit QNode must contain exactly one operation: an instance of `qp.evolve` acting on $|0\rangle$ evaluated at $t = 2$:
```python
@qp.qnode(dev)
def circuit(params):
    qp.evolve(hamiltonian())(params, t=t)
    return qp.state()
```

### 3. Fidelity Cost Function & JAX Autodiff
The fidelity between pure statevectors $|\psi_{\text{target}}\rangle$ and $|\psi(t)\rangle$ is calculated via:
$$F(|\psi_{\text{target}}\rangle, |\psi(t)\rangle) = |\langle \psi_{\text{target}} | \psi(t) \rangle|^2$$
Minimizing $-F$ allows gradient descent updates using JAX automatic differentiation (`jax.grad`):
```python
def cost(params):
    return -qp.math.fidelity_statevector(target_state(state), circuit(params))
```

---

## 3. Challenge Code Blocks

### Block 1: `hamiltonian()`
```python
def hamiltonian():
    """Creates the time-dependent Hamiltonian: H(t) = p0 t X + p1 t^2 Y + p2 t^3 Z
    
    Returns:
    (qp.pulse.parametrized_hamiltonian.ParametrizedHamiltonian): the Hamiltonian.
    """
    def f0(params, t):
        return params * t

    def f1(params, t):
        return params * (t ** 2)

    def f2(params, t):
        return params * (t ** 3)

    return f0 * qp.PauliX(0) + f1 * qp.PauliY(0) + f2 * qp.PauliZ(0)
```

### Block 2: `circuit(params)`
```python
@qp.qnode(dev)
def circuit(params):
    """A QNode that evolves the |0> state under the time-dependent Hamiltonian.
    This QNode must contain one operation only: an instance of qp.evolve.
    
    Args:
    params (list(numpy.tensor)): The parameters p0, p1, and p2 (in that order).
    Returns:
    (numpy.tensor): The evolved state.
    """
    qp.evolve(hamiltonian())(params, t=t)
    return qp.state()
```

### Block 3: `cost(params)` inside `optimal_parameters(state)`
```python
    def cost(params):
        """Computes the negative fidelity between the output of circuit and the target state
        
        args: 
            params (list(numpy.tensor)): The parameters p0, p1, and p2 (in that order).
        returns:
            (numpy.tensor): the negative fidelity.
        """
        return -qp.math.fidelity_statevector(target_state(state), circuit(params))
```

---

## 4. Verification & Results

```
Running test case 0 with input '[0.11775969, 0.99304212]'...
Correct!
Running test case 1 with input '[0.7071067812, 0.7071067812]'...
Correct!
```
