# Challenge: Break the Tie (200 pts)

## 1. Problem Overview
In quantum Monte Carlo (QMC), amplitude estimation is used to estimate the expectation value of a function $f(X)$ over a discrete probability distribution $P(X = i) = p_i$ on $2^m$ outcomes:
$$\mathbb{E}[f(X)] = \sum_{i=0}^{2^m - 1} p_i f_i$$

While classical Monte Carlo estimates this average via random sampling with an error convergence of $\mathcal{O}(1/\sqrt{N})$, quantum amplitude estimation achieves a quadratic speedup of $\mathcal{O}(1/N)$ in terms of circuit applications.

However, amplitude estimation introduces an inherent **two-fold ambiguity** in the measurement readout. Phase estimation applied to the Grover-like operator $Q$ yields two symmetric peaks in the estimation register:
- Peak at integer bin $k \approx 2^n \theta$
- Mirror peak at bin $2^n - k \approx 2^n (1 - \theta)$

These mirror peaks map to two indistinguishable candidate amplitudes:
$$a_1 = \frac{1 - \cos(\pi \theta)}{2}, \qquad a_2 = \frac{1 + \cos(\pi \theta)}{2} = 1 - a_1$$

Because both eigenvectors have equal overlap with the prepared state, the two peaks have equal height. No number of additional estimation wires can resolve this structural symmetry. 

The goal of this challenge is to:
1. Rescale $f_i$ into the $[0, 1]$ interval via $\tilde{f}_i = f_i / f_{\max}$.
2. Run `qp.QuantumMonteCarlo` on $m + 1 + n$ wires to obtain the two high-precision candidates $a_1$ and $a_2$.
3. Construct a shallow, direct tie-breaker circuit on $m + 1$ wires to measure $a_{\text{direct}} = \operatorname{Pr}[1] = \sum_i p_i \tilde{f}_i$ directly.
4. Pick whichever of $a_1$ or $a_2$ is closest to $a_{\text{direct}}$, and return the discretized estimate $f_{\max} \cdot a_{\text{chosen}}$.

---

## 2. Mathematical Derivation & Ambiguity Analysis

### 1. State Encoding
Given $2^m$ outcomes ($m$ data wires) and 1 objective wire (wire $m$):
1. **Distribution Loading ($A$):**
   $$A |0\rangle^{\otimes m} = \sum_{i=0}^{2^m - 1} \sqrt{p_i} |i\rangle$$
2. **Controlled Rotation ($R$):**
   For each outcome $i$, an $R_Y(\theta_i)$ rotation is applied to the objective wire controlled on data register $|i\rangle$, where:
   $$\theta_i = 2 \arcsin(\sqrt{\tilde{f}_i}), \qquad \tilde{f}_i = \frac{f_i}{f_{\max}}$$
   The resulting $(m+1)$-qubit state is:
   $$|\Psi\rangle = \sum_{i=0}^{2^m - 1} \sqrt{p_i} |i\rangle \left( \sqrt{1 - \tilde{f}_i} |0\rangle + \sqrt{\tilde{f}_i} |1\rangle \right)$$

Measuring the objective wire in the computational basis yields:
$$\operatorname{Pr}[1] = \sum_{i=0}^{2^m - 1} p_i \tilde{f}_i = \frac{\mathbb{E}[f(X)]}{f_{\max}} \equiv a$$

### 2. Origin of the Two-Fold Ambiguity
Amplitude estimation applies Quantum Phase Estimation (QPE) with $n$ estimation wires to the Grover-like operator $Q = - A S_0 A^\dagger S_\chi$.

In the 2D invariant subspace spanned by the "good" state ($|1\rangle$ on objective) and "bad" state ($|0\rangle$ on objective), $Q$ acts as a rotation by $2\phi$:
$$Q |\psi_\pm\rangle = e^{\pm 2\pi i \theta} |\psi_\pm\rangle$$
where the amplitude $a$ is related to the phase $\theta$ by:
$$a = \sin^2(\pi \theta) = \frac{1 - \cos(2\pi \theta)}{2} \quad \text{or in the challenge's normalization: } a = \frac{1 - \cos(\pi \theta)}{2}$$

The initial state $|\Psi\rangle$ has equal projections onto both eigenvectors $|\psi_+\rangle$ and $|\psi_-\rangle$. Consequently, QPE produces two peaks of identical height:
- Bin $k = \operatorname{round}(2^n \theta)$
- Mirror bin $2^n - k = \operatorname{round}(2^n (1 - \theta))$

Using $\cos(\pi(1 - \theta)) = -\cos(\pi \theta)$, these two bins yield the candidate pair:
$$a_1 = \frac{1 - \cos(\pi \theta)}{2}, \qquad a_2 = \frac{1 + \cos(\pi \theta)}{2} = 1 - a_1$$

### 3. Why Standard Docstring Recipes Fail
The default PennyLane `QuantumMonteCarlo` documentation example extracts the phase via:
```python
phase_estimated = np.argmax(circuit()[:int(N / 2)]) / N
```
This forces the search into the lower half $[0, 2^{n-1}]$, which assumes $a \le 0.5$ and always selects $\min(a_1, a_2)$. 
When the true expectation has $a > 0.5$ (such as in Test Case 2 where $a = 0.875$), this naive recipe outputs $0.125432$ instead of the correct $0.874568$.

### 4. Resolving the Ambiguity via Direct Circuit
To break the tie, we need an independent estimate of $a$ with error strictly less than $|0.5 - a|$. 
We build the direct encoding on $m + 1$ wires:
1. `qp.StatePrep(np.sqrt(probabilities), wires=range(m))`
2. Multi-controlled $R_Y(\theta_i)$ on wire $m$ controlled on data wires matching the bitstring of $i$:
   $$\theta_i = \text{rotation\_angle}(\tilde{f}_i) = 2 \arcsin(\sqrt{\tilde{f}_i})$$
3. Measuring `qp.probs(wires=[m])` returns $[\operatorname{Pr}[0], \operatorname{Pr}[1]]$, directly giving $a_{\text{direct}} = \operatorname{Pr}[1]$.

Comparing $|a_1 - a_{\text{direct}}|$ with $|a_2 - a_{\text{direct}}|$ unambiguously identifies the correct candidate $a_{\text{chosen}}$.
Finally, the estimated expectation value is:
$$\widehat{\mathbb{E}}[f(X)] = f_{\max} \cdot a_{\text{chosen}}$$

---

## 3. Implementation

```python
import json
import pennylane as qp
import pennylane.numpy as np

def rotation_angle(fraction):
    return float(2 * np.arcsin(np.sqrt(fraction)))

def run_circuit(num_wires, circuit):
    device = qp.device("default.qubit", wires=num_wires)
    return [float(value) for value in np.asarray(qp.QNode(circuit, device)()).ravel()]

def estimate_mean(probabilities, values, n):
    probabilities = np.array(probabilities, dtype=float)
    values = np.array(values, dtype=float)
    m = int(round(np.log2(len(probabilities))))     # number of data wires

    # --- Step 1: rescale the quantity into [0, 1] ----------------------------
    f_max = float(np.max(values))
    if f_max == 0.0:
        return 0.0
    f_rescaled = values / f_max

    # --- Step 2: run the amplitude estimation --------------------------------
    def amp_est_circuit():
        qp.QuantumMonteCarlo(
            probabilities,
            lambda i: f_rescaled[i],
            target_wires=range(m + 1),
            estimation_wires=range(m + 1, m + 1 + n),
        )
        return qp.probs(wires=range(m + 1, m + 1 + n))

    readout = run_circuit(m + 1 + n, amp_est_circuit)

    # --- Step 3: turn the readout into the two candidates --------------------
    k = int(np.argmax(readout))
    theta = k / (2**n)
    a1 = (1.0 - np.cos(np.pi * theta)) / 2.0
    a2 = (1.0 + np.cos(np.pi * theta)) / 2.0

    # --- Step 4: build the tie-breaker circuit  ------------------------------
    def tie_breaker_circuit():
        if m > 0:
            qp.StatePrep(np.sqrt(probabilities), wires=range(m))
            for i in range(2**m):
                angle = rotation_angle(f_rescaled[i])
                ctrl_vals = [(i >> (m - 1 - bit)) & 1 for bit in range(m)]
                qp.ctrl(qp.RY(angle, wires=m), control=range(m), control_values=ctrl_vals)
            return qp.probs(wires=[m])
        else:
            qp.RY(rotation_angle(f_rescaled[0]), wires=0)
            return qp.probs(wires=[0])

    direct_probs = run_circuit(m + 1, tie_breaker_circuit)
    a_direct = direct_probs[1]

    # --- Step 5: break the tie -----------------------------------------------
    a_chosen = a1 if abs(a1 - a_direct) < abs(a2 - a_direct) else a2
    return float(f_max * a_chosen)
```

---

## 4. Verification & Results

### Public Test Case 1:
- **Input:** Exponentially decaying distribution on 8 outcomes ($m=3$), quadratic values $f = [0, 0.25, 1.0, 2.25, 4.0, 6.25, 9.0, 12.25]$, $n=8$.
- **Expected:** `1.160949660702857`
- **Obtained:** `1.160949660702857` (Difference: `0.0`)
- **Outcome:** Smaller candidate picked ($a < 0.5$).

### Public Test Case 2:
- **Input:** Uniform distribution on 8 outcomes ($m=3$), values $f = [0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0]$, $n=8$.
- **Expected:** `0.8745681972617296`
- **Obtained:** `0.8745681972617296` (Difference: `0.0`)
- **Outcome:** Larger candidate picked ($a > 0.5$, resolving the tie correctly where naive lower-half slicing fails).
