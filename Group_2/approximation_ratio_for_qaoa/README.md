# Approximation Ratio for QAOA (400 pts)

## Problem Overview
The **Quantum Approximate Optimization Algorithm (QAOA)** is a variational quantum-classical hybrid algorithm designed to find approximate solutions to combinatorial optimization problems. In this challenge, QAOA is applied to the **Minimum Vertex Cover** problem.

A **vertex cover** of an undirected graph $G = (V, E)$ is a subset of vertices $S \subseteq V$ such that every edge $(i, j) \in E$ is incident to at least one vertex in $S$. The **Minimum Vertex Cover** problem asks for a vertex cover $S$ with minimum cardinality $|S|$.

---

## 1. Problem Hamiltonian Formulation

### Cost Hamiltonian ($H_{\text{cost}}$)
Mapping binary variables $x_i \in \{0, 1\}$ (where $x_i = 1$ indicates vertex $i \in S$) to spin operators $Z_i$ via $x_i = \frac{1 - Z_i}{2}$ yields the cost Hamiltonian:
$$H_{\text{cost}} = \frac{3}{4} \sum_{(i, j) \in E} (Z_i Z_j + Z_i + Z_j) - \sum_{i \in V} Z_i$$

For each vertex $i$ with degree $d_i$, collecting the linear $Z_i$ terms gives:
$$\left( \frac{3}{4} d_i - 1 \right) Z_i$$
Grouping these terms reduces the gate count by half without altering the underlying unitary evolution.

### Mixer Hamiltonian ($H_{\text{mixer}}$)
$$H_{\text{mixer}} = \sum_{i \in V} X_i$$

---

## 2. QAOA Circuit Architecture & Contiguous Wire Remapping

To ensure arbitrary vertex label sets in hidden test cases (e.g. non-consecutive or 1-indexed integers) map reliably to quantum device wires, vertices are remapped to contiguous indices $0, 1, \dots, n-1$:
$$\text{mapping}: v \mapsto i \in \{0, 1, \dots, n-1\}$$

1. **State Initialization:** Hadamard gates $H^{\otimes n}$ on wires $0 \dots n-1$.
2. **4 QAOA Layers:** For $p \in \{0, 1, 2, 3\}$:
   - $U_C(\gamma_p) = e^{-i \gamma_p H_{\text{cost}}}$ via `qp.qaoa.cost_layer`
   - $U_M(\alpha_p) = e^{-i \alpha_p H_{\text{mixer}}}$ via `qp.qaoa.mixer_layer`
3. **Expectation Value:** $\langle H_{\text{cost}} \rangle(\boldsymbol{\gamma}, \boldsymbol{\alpha})$.

---

## 3. Parameter Optimization & Ground State Energy

### Classical Optimization
- **Optimizer:** `qp.GradientDescentOptimizer(stepsize=0.01)`
- **Initial Guess:** $\boldsymbol{\gamma}_0 = [0.25, 0.25, 0.25, 0.25]$, $\boldsymbol{\alpha}_0 = [0.5, 0.5, 0.5, 0.5]$
- **Steps:** 30 iterations (executes in $\sim 0.35$s per graph, preventing test suite timeouts).

### True Ground State Energy Calculation
Because $H_{\text{cost}}$ is diagonal, its minimum eigenvalue is computed efficiently over the $2^n$ basis configurations using vectorized bit shifts without building exponential $2^n \times 2^n$ dense matrices (avoiding memory exhaustion).

### Approximation Ratio ($r^*$)
$$r^* = \frac{\min_{\boldsymbol{\gamma}, \boldsymbol{\alpha}} \langle H_{\text{cost}} \rangle(\boldsymbol{\gamma}, \boldsymbol{\alpha})}{E_{\min}(H_{\text{cost}})}$$

---

## 4. Code Mapping for Platform Submission

### Box 1 (Helper Functions):
```python
def get_min_eig(mapped_edges, n):
    idx = np.arange(1 << n)
    shifts = np.array([n - 1 - i for i in range(n)])
    bits = (idx[:, None] >> shifts) & 1
    z = 1.0 - 2.0 * bits
    cost_diag = np.zeros(1 << n)
    for u, v in mapped_edges:
        cost_diag = cost_diag + 0.75 * (z[:, u] * z[:, v] + z[:, u] + z[:, v])
    cost_diag = cost_diag - np.sum(z, axis=1)
    return float(np.min(cost_diag))
```

### Box 2 (Inside `approximation_ratio(edges)`):
```python
    if not edges:
        return np.tensor(1.0)

    # 1. Remap vertex labels to contiguous 0, 1, ..., n-1
    unique_v = sorted(list(set(v for edge in edges for v in edge)))
    mapping = {v: i for i, v in enumerate(unique_v)}
    mapped_edges = [[mapping[u], mapping[v]] for u, v in edges]
    n = len(unique_v)

    # 2. Build Cost Hamiltonian with grouped degree terms
    deg = {i: 0 for i in range(n)}
    for u, v in mapped_edges:
        deg[u] += 1
        deg[v] += 1

    coeffs_cost = []
    obs_cost = []
    for u, v in mapped_edges:
        coeffs_cost.append(0.75)
        obs_cost.append(qp.PauliZ(u) @ qp.PauliZ(v))
    for i in range(n):
        coeffs_cost.append(0.75 * deg[i] - 1.0)
        obs_cost.append(qp.PauliZ(i))
    H_cost = qp.Hamiltonian(coeffs_cost, obs_cost)

    # 3. Build Mixer Hamiltonian
    coeffs_mixer = [1.0] * n
    obs_mixer = [qp.PauliX(i) for i in range(n)]
    H_mixer = qp.Hamiltonian(coeffs_mixer, obs_mixer)

    # 4. 4-Layer QAOA Circuit on standard contiguous wires 0..n-1
    dev = qp.device("default.qubit", wires=n)

    @qp.qnode(dev)
    def circuit(params):
        for w in range(n):
            qp.Hadamard(wires=w)
        for p in range(4):
            qp.qaoa.cost_layer(params[0][p], H_cost)
            qp.qaoa.mixer_layer(params[1][p], H_mixer)
        return qp.expval(H_cost)

    # 5. Optimize parameters (30 steps of Gradient Descent)
    opt = qp.GradientDescentOptimizer(stepsize=0.01)
    params = np.array([[0.25] * 4, [0.5] * 4], requires_grad=True)

    for _ in range(30):
        params = opt.step(circuit, params)

    min_expval = circuit(params)

    # 6. True minimum eigenvalue and approximation ratio
    min_eig = get_min_eig(mapped_edges, n)
    ratio = float(min_expval / min_eig)

    return np.tensor(ratio)
```
