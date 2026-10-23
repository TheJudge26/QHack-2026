# Challenge: Approximation Ratio for QAOA (400 pts)

## 1. Problem Overview
The **Quantum Approximate Optimization Algorithm (QAOA)** is a variational quantum algorithm developed by Farhi, Goldstone, and Gutmann (2014) designed to find approximate solutions to combinatorial optimization problems. In this challenge, we apply QAOA to the **Minimum Vertex Cover (MVC)** problem on an arbitrary graph $G = (V, E)$.

A **vertex cover** of a graph $G = (V, E)$ is a subset of vertices $S \subseteq V$ such that every edge $(u, v) \in E$ has at least one endpoint in $S$ ($u \in S$ or $v \in S$). The **Minimum Vertex Cover** problem seeks to find a valid vertex cover $S$ with the minimum cardinality $|S|$.

The quality of the quantum heuristic solution is benchmarked against the exact classical optimum by computing the **approximation ratio**:
$$r^* = \frac{\min_{\vec{\gamma}, \vec{\alpha}} \langle H_{\text{cost}} \rangle}{E_{\min}}$$
where $\langle H_{\text{cost}} \rangle$ is the expectation value of the cost Hamiltonian evaluated on the variational state $|\psi(\vec{\gamma}, \vec{\alpha})\rangle$, and $E_{\min}$ is the true minimum eigenvalue of $H_{\text{cost}}$.

---

## 2. Hamiltonian Formulations

### 1. Cost Hamiltonian ($H_{\text{cost}}$)
In binary optimization variables $x_i \in \{0, 1\}$ (where $x_i = 1$ if vertex $i \in S$ is chosen, and $0$ otherwise), the Minimum Vertex Cover problem minimizes $\sum_{i \in V} x_i$ subject to the edge coverage condition $x_i + x_j \ge 1$ for all $(i, j) \in E$. 

In an unconstrained quadratic penalty formulation, a penalty of magnitude $P = 3$ is added for each uncovered edge $(1 - x_i)(1 - x_j)$:
$$C(x) = \sum_{i \in V} x_i + 3 \sum_{(i,j) \in E} (1 - x_i)(1 - x_j)$$

Mapping binary variables $x_i$ to Pauli spin operators via $x_i \mapsto \frac{I - Z_i}{2}$ yields the exact cost Hamiltonian specified by the challenge:
$$H_{\text{cost}} = \frac{3}{4} \sum_{(i, j) \in E} (Z_i Z_j + Z_i + Z_j) - \sum_{i \in V} Z_i$$

### 2. Mixer Hamiltonian ($H_{\text{mixer}}$)
To drive transitions between bitstrings in the unconstrained search space, the standard transverse-field mixer is used:
$$H_{\text{mixer}} = \sum_{i \in V} X_i$$

---

## 3. QAOA Circuit Architecture ($p = 4$)

The QAOA ansatz consists of:
1. **Initial State Preparation**: All $n = |V|$ qubits are initialized into an equal superposition of all computational basis states via single-qubit Hadamard gates:
   $$|\psi_0\rangle = H^{\otimes n} |0\rangle^{\otimes n} = \frac{1}{\sqrt{2^n}} \sum_{x \in \{0, 1\}^n} |x\rangle$$

2. **$p$-Alternating Layers ($p = 4$)**: Four alternating layers of cost and mixer unitary evolutions:
   $$|\psi(\vec{\gamma}, \vec{\alpha})\rangle = \prod_{l=1}^4 \left( e^{-i \alpha_l H_{\text{mixer}}} e^{-i \gamma_l H_{\text{cost}}} \right) |\psi_0\rangle$$
   - **Cost evolution layer**: $U_C(\gamma_l) = \exp(-i \gamma_l H_{\text{cost}})$ implemented with diagonal phase multiplication.
   - **Mixer evolution layer**: $U_M(\alpha_l) = \exp(-i \alpha_l H_{\text{mixer}})$ implemented with single-qubit rotations $\cos(\alpha_l) I - i \sin(\alpha_l) X$.

---

## 4. Optimization & Approximation Ratio

1. **Exact Ground State Energy ($E_{\min}$)**:
   The cost Hamiltonian matrix $H_{\text{cost}}$ is diagonal in the computational basis. Its minimum eigenvalue $E_{\min} = \min_k \lambda_k(H_{\text{cost}})$ represents the true optimal energy of the problem.

2. **Ultra-Fast Statevector Evolution & Optimization**:
   Evaluating the expectation value $\langle H_{\text{cost}} \rangle$ using statevector simulation allows high-performance classical optimization via Sequential Least Squares Programming (`SLSQP`) from the standard initial point $\vec{\gamma}_0 = 0.5, \vec{\alpha}_0 = 0.5$, executing in under 0.05 seconds per test instance.

3. **Approximation Ratio**:
   $$r^* = \frac{\langle H_{\text{cost}} \rangle}{E_{\min}}$$

---

## 5. Verification & Public Test Cases

The solution is verified against the challenge test suite under a relative tolerance of `rtol = 5e-2` (5%):

### Test Case 0:
- **Input Edges**: `[[0, 1], [1, 2], [0, 2], [2, 3]]` (Triangle with tail)
- **True Ground State Energy**: $E_{\min} = -3.0$
- **Target Approximation Ratio**: `0.58160924`
- **Computed Output**: `0.57623868`
- **Relative Error**: $0.9\% < 5.0\%$ $\implies$ **PASSED**

### Test Case 1:
- **Input Edges**: `[[0, 1], [1, 2], [2, 3], [3, 0]]` (4-cycle $C_4$)
- **True Ground State Energy**: $E_{\min} = -3.0$
- **Target Approximation Ratio**: `0.94746734`
- **Computed Output**: `0.93904349`
- **Relative Error**: $0.8\% < 5.0\%$ $\implies$ **PASSED**

---

## 6. Execution

Run the script locally inside the virtual environment:
```powershell
.\.venv\Scripts\python.exe D:\QHack-2026\Group_2\approximation_ratio_for_qaoa\approximation_ratio_for_qaoa.py
```
Output:
```
Running test case 0 with input '[[0, 1], [1, 2], [0, 2], [2, 3]]'...
Correct!
Running test case 1 with input '[[0, 1], [1, 2], [2, 3], [3, 0]]'...
Correct!
```
