# Challenge: The Draft Lottery (200 pts)

## 1. Problem Overview
In practical Quantum Phase Estimation (QPE), target states rarely match an exact single eigenvector. Instead, they are superpositions over multiple eigenvectors:
$$|\psi\rangle = \sum_k c_k |u_k\rangle$$
where $U|u_k\rangle = e^{2\pi i \theta_k}|u_k\rangle$.

By the linearity of quantum mechanics:
$$|\psi\rangle |0\rangle^{\otimes m} \xrightarrow{\text{QPE}} \sum_k c_k |u_k\rangle |\tilde{\theta}_k\rangle$$

Measuring the estimation register yields bitstring $\tilde{\theta}_k$ with probability:
$$\Pr[\tilde{\theta}_k] = |c_k|^2$$

This challenge demonstrates two concepts:
1. **Reading the Spectrum**: Collecting every bin with probability $> 10^{-9}$ recovers all distinct eigenphases and their weights.
2. **Pricing Repetitions**: Observing an eigenphase of weight $w$ at least once with $\ge 99\%$ confidence requires:
   $$(1 - w)^r \le 0.01 \implies r \ge \frac{\ln(0.01)}{\ln(1 - w)}$$
   rounded up. A faint eigenphase ($w = 0.01$) dramatically increases runtime ($r = 459$).

---

## 2. Circuit & Algorithm Architecture

### Registers and Wires
* $t = \log_2(\text{len}(phases))$ target wires: `range(t)`
* $m$ estimation wires: `range(t, t + m)`
* Total wires: $t + m$

### Execution Flow
1. **State Preparation**: `qp.StatePrep(np.array(amplitudes), wires=target_wires)`
2. **Phase Estimation**: `qp.QuantumPhaseEstimation(U, target_wires=target_wires, estimation_wires=estimation_wires)`
3. **Measurement**: `qp.probs(wires=estimation_wires)`

### Peak Detection & Analysis
* For each bin $k \in \{0, \dots, 2^m - 1\}$ where probability $> 10^{-9}$:
  $$\text{phase} = \frac{k}{2^m}$$
  Accumulate weight: $\text{peaks}[\text{phase}] += \text{weight}$
* Sort eigenphases ascending, pair with weights.
* Determine $w_{\text{faint}} = \min(\text{weights})$ and compute:
  $$r = \left\lceil \frac{\ln(0.01)}{\ln(1 - w_{\text{faint}})} \right\rceil$$

---

## 3. Implementation
```python
def read_spectrum(phases, amplitudes, m):
    n_target = int(np.log2(len(phases)))
    target_wires = list(range(n_target))
    estimation_wires = list(range(n_target, n_target + m))
    total_wires = n_target + m

    dev = qp.device("default.qubit", wires=total_wires)
    U = unitary_matrix(phases)

    @qp.qnode(dev)
    def circuit():
        qp.StatePrep(np.array(amplitudes), wires=target_wires)
        qp.QuantumPhaseEstimation(U, target_wires=target_wires, estimation_wires=estimation_wires)
        return qp.probs(wires=estimation_wires)

    distribution = circuit()

    peaks = {}
    for k, weight in enumerate(distribution):
        if weight > 1e-9:
            phase = k / (2 ** m)
            peaks[phase] = peaks.get(phase, 0.0) + float(weight)

    eigenphases = sorted(peaks.keys())
    weights = [peaks[p] for p in eigenphases]
    faintest = min(weights)
    repetitions = int(np.ceil(np.log(0.01) / np.log(1.0 - faintest)))

    return eigenphases, weights, repetitions
```
