# Approximation Ratio for QAOA (400 pts)

## Problem Overview
The Quantum Approximate Optimization Algorithm (QAOA) is a variational quantum algorithm designed to find near-optimal solutions to combinatorial optimization problems.

In this challenge, QAOA is applied to the **Minimum Vertex Cover (MVC)** problem on an arbitrary graph $G = (V, E)$. A vertex cover is a subset of vertices $S \subseteq V$ such that every edge $(i, j) \in E$ has at least one endpoint in $S$. The goal is to minimize $|S|$.

---

## 1. Problem Formulation & Hamiltonians
Mapping binary variables $x_i \in \{0, 1\}$ (where $x_i = 1$ if vertex $i \in S$) to spin variables via $x_i = \frac{1 - Z_i}{2}$:

- **Cost Hamiltonian**:
  $$H_{\text{cost}} = \frac{3}{4} \sum_{(i, j) \in E} (Z_i Z_j + Z_i + Z_j) - \sum_{i \in V} Z_i$$
  where the penalty term penalizes uncovered edges ($x_i = x_j = 0$) and the linear term minimizes the total number of vertices chosen.

- **Mixer Hamiltonian**:
  $$H_{\text{mixer}} = \sum_{i \in V} X_i$$
  which drives transitions between computational basis states.

---

## 2. QAOA Circuit (Depth $p=4$)
1. **Initialization**: Prepare the equal superposition state:
   $$|\psi_0\rangle = H^{\otimes |V|} |0\rangle^{\otimes |V|}$$
2. **Alternating Unitaries**: Apply 4 layers of alternating cost and mixer evolutions:
   $$|\psi(\gamma, \beta)\rangle = \prod_{l=1}^{p=4} \left( e^{-i \beta_l H_{\text{mixer}}} e^{-i \gamma_l H_{\text{cost}}} \right) |\psi_0\rangle$$
   In PennyLane:
   ```python
   def qaoa_layer(gamma, beta):
       qp.qaoa.cost_layer(gamma, H_cost)
       qp.qaoa.mixer_layer(beta, H_mixer)

   qp.layer(qaoa_layer, 4, params[0], params[1])
   ```

---

## 3. Approximation Ratio
The performance of QAOA is quantified by the approximation ratio:
$$r^* = \frac{\min_{\gamma, \beta} \langle \psi(\gamma, \beta) | H_{\text{cost}} | \psi(\gamma, \beta) \rangle}{\min \lambda(H_{\text{cost}})}$$

where $\min \lambda(H_{\text{cost}})$ is the exact classical minimum eigenvalue of the diagonal cost Hamiltonian.

Optimization is performed using PennyLane's `qp.AdamOptimizer(stepsize=0.1)` over 50 steps from initial angles $\gamma_l = 0.5, \beta_l = 0.5$.

---

## 4. Implementation
```python
def build_hamiltonians(edges):
    vertices = sorted(list(set(v for edge in edges for v in edge)))

    cost_coeffs = []
    cost_terms = []
    for i, j in edges:
        cost_coeffs.extend([0.75, 0.75, 0.75])
        cost_terms.extend([qp.PauliZ(i) @ qp.PauliZ(j), qp.PauliZ(i), qp.PauliZ(j)])

    for i in vertices:
        cost_coeffs.append(-1.0)
        cost_terms.append(qp.PauliZ(i))

    H_cost = qp.Hamiltonian(cost_coeffs, cost_terms)

    mixer_coeffs = [1.0] * len(vertices)
    mixer_terms = [qp.PauliX(i) for i in vertices]
    H_mixer = qp.Hamiltonian(mixer_coeffs, mixer_terms)

    return H_cost, H_mixer, vertices


def approximation_ratio(edges):
    H_cost, H_mixer, vertices = build_hamiltonians(edges)
    num_wires = len(vertices)
    dev = qp.device("default.qubit", wires=num_wires)

    depth = 4

    def qaoa_layer(gamma, beta):
        qp.qaoa.cost_layer(gamma, H_cost)
        qp.qaoa.mixer_layer(beta, H_mixer)

    @qp.qnode(dev)
    def circuit(params):
        for w in vertices:
            qp.Hadamard(wires=w)
        qp.layer(qaoa_layer, depth, params[0], params[1])
        return qp.expval(H_cost)

    opt = qp.AdamOptimizer(stepsize=0.1)
    params = np.array([[0.5] * depth, [0.5] * depth], requires_grad=True)

    for _ in range(50):
        params, cost = opt.step_and_cost(circuit, params)

    min_eig = np.min(qp.eigvals(H_cost))
    ratio = cost / min_eig

    return np.tensor(ratio)
```
