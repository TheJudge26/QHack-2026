# A Basket of Options (400 pts)

## Problem Overview
In quantitative finance, pricing derivative contracts on multiple underlying assets (e.g., a basket call option) requires evaluating high-dimensional expectations over correlated stochastic processes.

Classical Monte Carlo methods estimate expectations with a precision that scales as $\mathcal{O}(1/\sqrt{N_{\text{samples}}})$, requiring $\mathcal{O}(1/\epsilon^2)$ function evaluations for target error $\epsilon$. **Quantum Monte Carlo (QMC)**, based on Quantum Amplitude Estimation (QAE), achieves a quadratic speedup of $\mathcal{O}(1/\epsilon)$.

---

## 1. Asset Dynamics & Basket Payoff
The terminal prices $S_{1, T}$ and $S_{2, T}$ follow correlated geometric Brownian motions:
$$S_{i, T} = S_{i, 0} \exp\left[\left(r - \frac{\sigma_i^2}{2}\right)T + \sigma_i \sqrt{T} Z_i\right], \quad i \in \{1, 2\}$$

where $(Z_1, Z_2)^T \sim \mathcal{N}\left(\mathbf{0}, \begin{pmatrix} 1 & \rho \\ \rho & 1 \end{pmatrix}\right)$.

For a basket with weight $w \in [0, 1]$:
$$B_T = w S_{1, T} + (1 - w) S_{2, T}$$

The payoff of a European basket call with strike $K$ is:
$$g(S_{1, T}, S_{2, T}) = \max(B_T - K, 0)$$

By risk-neutral pricing:
$$V_0 = e^{-rT} \mathbb{E}[g(S_{1, T}, S_{2, T})] = e^{-rT} g_{\max} \mu$$
where $\mu = \mathbb{E}[f] = \sum_i p_i f(i)$ is the expectation of the normalized payoff $f(i) = \frac{g_i}{g_{\max}} \in [0, 1]$.

---

## 2. Scenario Discretization
The shocks $(Z_1, Z_2)$ are discretized on $[-3, 3] \times [-3, 3]$ using an $8 \times 8$ grid ($M = 64$ scenarios, $m = \log_2(64) = 6$ qubits).
For each pair $(z_1, z_2)$:
$$p(z_1, z_2) = \frac{1}{2\pi \sqrt{1 - \rho^2}} \exp\left(-\frac{z_1^2 - 2\rho z_1 z_2 + z_2^2}{2(1 - \rho^2)}\right)$$
normalized such that $\sum_{i=0}^{63} p_i = 1$.

The scenario index $i \in \{0, \dots, 63\}$ maps to:
$$z_1 = \text{grid}[i // 8], \quad z_2 = \text{grid}[i \% 8]$$

---

## 3. Quantum Monte Carlo Routine
PennyLane's `qp.templates.QuantumMonteCarlo` uses $m+1 = 7$ target wires (6 for scenarios, 1 ancilla) and $n = 10$ estimation wires ($N = 1024$):
1. **$\mathcal{A}$ operator**: Loads $\sum_{i=0}^{M-1} \sqrt{p_i} |i\rangle$ on the $m$ scenario qubits.
2. **$\mathcal{R}$ operator**: Controlled-$Y$ rotation on the ancilla qubit encoding:
   $$\sqrt{1 - f(i)} |0\rangle + \sqrt{f(i)} |1\rangle$$
3. **$\mathcal{Q}$ Grover operator**: Defines reflection unitaries with eigenvalues $e^{\pm 2\pi i \theta}$ where $\mu = \frac{1 - \cos(\pi \theta)}{2}$.
4. **Quantum Phase Estimation (QPE)**: Estimates phase $\theta$ onto the $n = 10$ estimation wires.

The estimated phase $\phi = \frac{\text{argmax}(P)}{N}$ yields:
$$V_0 = e^{-rT} g_{\max} \left(\frac{1 - \cos(\pi \phi)}{2}\right)$$

---

## 4. Implementation
```python
# Helper functions
grid = np.linspace(-3, 3, 8)

def get_probs(rho):
    probs = []
    norm = 1.0 / (2 * np.pi * np.sqrt(1 - rho**2))
    for z1 in grid:
        for z2 in grid:
            exponent = -(z1**2 - 2 * rho * z1 * z2 + z2**2) / (2 * (1 - rho**2))
            p = norm * np.exp(exponent)
            probs.append(p)
    probs = np.array(probs)
    return probs / np.sum(probs)

def payoff(i, w, T, K):
    z1 = grid[i // 8]
    z2 = grid[i % 8]
    S1_T = S10 * np.exp((r - 0.5 * sigma1**2) * T + sigma1 * np.sqrt(T) * z1)
    S2_T = S20 * np.exp((r - 0.5 * sigma2**2) * T + sigma2 * np.sqrt(T) * z2)
    B_T = w * S1_T + (1 - w) * S2_T
    return max(B_T - K, 0.0)

# Quantum circuit
@qp.qnode(dev)
def quantum_montecarlo(rho, w, T, K):
    probs = get_probs(rho)
    payoffs = np.array([payoff(i, w, T, K) for i in range(M)])
    g_max = np.max(payoffs)
    func = lambda i: payoff(i, w, T, K) / g_max

    qp.templates.QuantumMonteCarlo(
        probs,
        func,
        target_wires=target_wires,
        estimation_wires=estimation_wires,
    )

    return qp.probs(estimation_wires)
```
