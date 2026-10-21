# A Relay of Phases (200 pts)

## Problem Overview
Instantaneous Quantum Polynomial-time (IQP) circuits are quantum circuits generated entirely by mutually commuting gates. Despite their commuting structure, sampling from IQP circuits is classically intractable under standard complexity-theoretic assumptions (e.g., polynomial hierarchy collapse).

In this challenge, we construct and optimize an IQP circuit on $n=4$ qubits:
$$U(\theta) = \prod_{j} \exp\left(-i \frac{\theta_j}{2} X_{S_j}\right), \quad X_{S_j} = \prod_{k \in S_j} X_k$$

and find parameters $\theta$ that minimize the expectation value of a multi-qubit Pauli-$Z$ observable:
$$Z_a = \prod_{k=0}^{n-1} Z_k^{a_k}, \quad a_k \in \{0, 1\}$$
$$C(\theta) = \langle 0 | U^\dagger(\theta) Z_a U(\theta) | 0 \rangle$$

---

## 1. Circuit Construction
Each pattern group $j$ contains one or more qubit subsets $S_j$.
For each subset $S_j = [w_1, \dots, w_m]$, the rotation generator is the multi-qubit Pauli-$X$ string $X^{\otimes m}$.
In PennyLane, this is implemented using `qp.PauliRot`:
```python
qp.PauliRot(theta, 'X' * len(wires), wires=wires)
```
which evaluates to $\exp(-i \frac{\theta}{2} X_{S_j})$.

The observable $Z_a$ is built from the non-zero indices of $a$:
```python
obs_wires = [i for i, val in enumerate(observable) if val == 1]
obs = None
for w in obs_wires:
    obs = qp.PauliZ(w) if obs is None else obs @ qp.PauliZ(w)
```

---

## 2. Theoretical Analysis & Energy Landscape
Because all generators $X_{S_j}$ consist strictly of Pauli-$X$ operators, all gates in $U(\theta)$ commute.
When conjugating the Pauli-$Z$ observable $Z_a$:
1. If $X_{S_j}$ and $Z_a$ commute (even overlap of active wires), $X_{S_j}$ has no effect on $Z_a$.
2. If $X_{S_j}$ and $Z_a$ anticommute (odd overlap of active wires):
   $$\exp\left(i \frac{\theta_j}{2} X_{S_j}\right) Z_a \exp\left(-i \frac{\theta_j}{2} X_{S_j}\right) = \cos(\theta_j) Z_a + \sin(\theta_j)(i X_{S_j} Z_a)$$

Evaluating in the reference ground state $|0\rangle^{\otimes n}$ (where $\langle 0 | Z_a | 0 \rangle = 1$ and all Pauli-$Y$ cross terms vanish):
$$C(\theta) = \prod_{j \in \mathcal{A}} \cos(\theta_j)$$
where $\mathcal{A}$ is the set of parameter groups that anticommute with $Z_a$.

### The Symmetry Trap
If initialized symmetrically ($\theta_j = \theta$), then for $|\mathcal{A}| = 2$:
$$C(\theta) = \cos^2(\theta) \ge 0$$
The landscape has a saddle/local minimum at $\theta = \pi/2$ where $C = 0$. Gradient descent initialized symmetrically cannot reach the global minimum $-1.0$.

### Exact Global Minimizer
By setting any single parameter $j \in \mathcal{A}$ to $\pi$ and all other parameters to $0$:
$$C(\theta) = \cos(\pi) \cdot \prod_{k \neq j} \cos(0) = (-1) \cdot 1 = -1$$
This achieves the theoretical global minimum of $-1.0$ exactly.

---

## 3. Implementation
```python
@qp.qnode(dev)
def iqp_circuit(weights, pattern, observable):
    for theta, group in zip(weights, pattern):
        for wires in group:
            qp.PauliRot(theta, 'X' * len(wires), wires=wires)

    obs_wires = [i for i, val in enumerate(observable) if val == 1]
    if not obs_wires:
        obs = qp.Identity(0)
    else:
        obs = None
        for w in obs_wires:
            obs = qp.PauliZ(w) if obs is None else obs @ qp.PauliZ(w)

    return qp.expval(obs)


def optimize_iqp(pattern, observable):
    # Direct check for parameter setting to pi (exact -1 minimizer)
    for i in range(len(pattern)):
        w = np.zeros(len(pattern), requires_grad=True)
        w[i] = np.pi
        if np.isclose(float(iqp_circuit(w, pattern, observable)), -1.0, atol=1e-3):
            return w

    # Fallback to Adam optimizer
    weights = np.random.uniform(0.1, np.pi, len(pattern), requires_grad=True)
    opt = qp.AdamOptimizer(stepsize=0.1)
    for _ in range(100):
        weights, c = opt.step_and_cost(lambda w: iqp_circuit(w, pattern, observable), weights)
        if c <= -0.99:
            break
    return weights
```
