# Challenge: The Salary Cap (200 pts)

## 1. Problem Overview

In quantum resource estimation, the traditional pipeline begins with a specified quantum circuit, compiles it into a target fault-tolerant gate set (e.g., Clifford + T), counts the non-Clifford $T$ gates $N_T$, and computes the expected wall-clock execution time:
$$T_{\text{wall}} = N_T \cdot t_T$$
where $t_T$ is the execution time of a single logical $T$ gate (governed by error-correcting code distance and physical cycle time).

In practical quantum systems engineering, the question is often inverted:
> Given hardware performance constraints ($t_T$) and a fixed wall-clock computing budget ($T_{\text{budget}}$), what is the largest instance size of an algorithm that can be executed within budget?

### Quantum Phase Estimation (QPE) Scaling
Quantum Phase Estimation (QPE) provides a clean testbed for this resource trade-off:
- **Phase Resolution:** With $m$ estimation wires, the phase resolution is:
  $$\epsilon = 2^{-m}$$
  Each additional estimation wire halves the uncertainty (adds 1 bit of precision).
- **Runtime Trade-off:** QPE applies controlled powers of the base unitary $U$:
  $$\sum_{k=0}^{m-1} 2^k = 2^m - 1$$
  For a generic multi-qubit base unitary (such as a 2-qubit unitary without power-collapsing structure), the number of base unitary applications doubles with each added qubit. Thus, each additional bit of precision costs roughly twice the execution time.
- **Rotation Synthesis:** Single-qubit rotations inside the inverse Quantum Fourier Transform (QFT) and unitary synthesis must be decomposed into Clifford+T sequences to precision $\epsilon_{\text{rot}} = 10^{-6}$, contributing an overhead proportional to $\mathcal{O}(\log(1/\epsilon_{\text{rot}}))$ $T$ gates per rotation.

The goal of this challenge is to determine the largest integer $m$ (number of estimation wires) such that:
$$T_{\text{wall}}(m) = N_T(m) \cdot t_T \le T_{\text{budget}}$$

---

## 2. Mathematical Derivation & Algorithmic Analysis

### 1. Symbolic Resource Estimation via `pennylane.estimator`
Resource estimation in PennyLane is performed symbolically via `qp.estimator` (`qre`), bypassing state-vector simulation to evaluate asymptotic gate counts directly:
1. **Target Gate Set:** The fault-tolerant Clifford+T basis:
   $$\mathcal{G} = \{ \text{Hadamard}, S, CNOT, T \}$$
2. **Rotation Precision:** Fixed to $\epsilon_{\text{rot}} = 10^{-6}$ via a `qre.ResourceConfig`:
   ```python
   config = qre.ResourceConfig()
   config.set_single_qubit_rot_precision(1e-6)
   ```
3. **Target Routine:** Controlled powers of a generic 2-qubit unitary `qre.QubitUnitary(num_wires=2)` inside `qre.QPE(..., num_estimation_wires=m)`.

### 2. Search Strategy
Because $N_T(m)$ is strictly monotonically increasing with $m$:
- We scan upward starting from $m = 1$:
  - Estimate resources using `qre.estimate(qre.QPE(qre.QubitUnitary(num_wires=2), num_estimation_wires=m), gate_set=GATE_SET, config=config)`.
  - Extract the exact $T$-count: $N_T(m) = \text{resources.gate\_counts.get}("T", 0)$.
  - Evaluate wall-clock runtime: $T_{\text{wall}} = N_T(m) \cdot t_T$.
  - If $T_{\text{wall}} \le T_{\text{budget}}$, record $m$ as the current best configuration:
    $$\text{best} = (m, 2^{-m}, N_T(m))$$
  - Once $T_{\text{wall}} > T_{\text{budget}}$, terminate immediately. The previous recorded size is guaranteed to be the maximal affordable precision.

---

## 3. Implementation

Located in [`the_salary_cap.py`](file:///d:/QHack-2026/Group_3/the_salary_cap/the_salary_cap.py):

```python
import json
import pennylane as qp
import pennylane.numpy as np

qre = qp.estimator

GATE_SET = {"Hadamard", "S", "CNOT", "T"}
ROTATION_PRECISION = 1e-6


def affordable_precision(t_gate_seconds, budget_seconds, max_estimation_wires=40):
    """Finds the largest quantum phase estimation that fits a wall-clock budget."""
    config = qre.ResourceConfig()
    config.set_single_qubit_rot_precision(ROTATION_PRECISION)

    best = None
    for m in range(1, max_estimation_wires + 1):
        resources = qre.estimate(
            qre.QPE(qre.QubitUnitary(num_wires=2), num_estimation_wires=m),
            gate_set=GATE_SET,
            config=config,
        )
        t_count = int(resources.gate_counts.get("T", 0))
        if t_count * t_gate_seconds <= budget_seconds:
            best = (int(m), float(2.0 ** -m), int(t_count))
        else:
            break

    return best
```

---

## 4. Verification & Validation

Running the automated test suite locally:

```bash
python Group_3/the_salary_cap/the_salary_cap.py
```

### Public Test Cases
1. **Case 1: $t_T = 10^{-6}\text{ s}$ ($1\,\mu\text{s}$), $T_{\text{budget}} = 1.0\text{ s}$:**
   - Budget: $10^6$ $T$-gates.
   - At $m = 9$: $N_T = 684,108 \implies T_{\text{wall}} = 0.6841\text{ s} \le 1.0\text{ s}$ (Affordable).
   - At $m = 10$: $N_T = 1,366,956 \implies T_{\text{wall}} = 1.3670\text{ s} > 1.0\text{ s}$ (Overruns).
   - **Result:** `(9, 0.001953125, 684108)` $\implies$ Correct!

2. **Case 2: $t_T = 10^{-3}\text{ s}$ ($1\text{ ms}$), $T_{\text{budget}} = 60.0\text{ s}$:**
   - Budget: $60,000$ $T$-gates.
   - At $m = 5$: $N_T = 42,252 \implies T_{\text{wall}} = 42.252\text{ s} \le 60.0\text{ s}$ (Affordable).
   - At $m = 6$: $N_T = 85,020 \implies T_{\text{wall}} = 85.020\text{ s} > 60.0\text{ s}$ (Overruns).
   - **Result:** `(5, 0.03125, 42252)` $\implies$ Correct!
