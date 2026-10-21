# Challenge: Goal or No Goal? (100 pts)

## 1. Problem Overview
The goal is to prepare an $n$-qubit GHZ state and uniquely certify it using **$n$ parity checks**:
1. One **global $X$ parity**: $\Pi_X = X_0 X_1 \cdots X_{n-1}$
2. $n - 1$ **neighbouring $ZZ$ parities**: $Z_i Z_{i+1}$ for $i \in \{0, 1, \dots, n-2\}$

The average of these $n$ checks:
$$s_n = \frac{1}{n} \left( \langle \Pi_X \rangle + \sum_{i=0}^{n-2} \langle Z_i Z_{i+1} \rangle \right)$$
equals **$1.0$ if and only if** the state is the true GHZ state:
$$|GHZ_n\rangle = \frac{1}{\sqrt{2}} (|00\dots 0\rangle + |11\dots 1\rangle)$$

---

## 2. Why Are Both $X$ and $Z$ Checks Necessary?

### Trap 1: Only measuring in the $Z$ basis (Bitstring histograms)
* If you measure each qubit in the computational basis, $|GHZ_n\rangle$ yields $|00\dots 0\rangle$ with probability $0.5$ and $|11\dots 1\rangle$ with probability $0.5$.
* However, a **classical statistical mixture** (e.g. from flipping a fair classical coin):
  $$\rho_{\text{mix}} = \frac{1}{2}|00\dots 0\rangle\langle 00\dots 0| + \frac{1}{2}|11\dots 1\rangle\langle 11\dots 1|$$
  produces the exact same bitstring histogram!
* The difference between quantum superposition and classical mixture lies in the **off-diagonal coherences**:
  $$\frac{1}{2}|00\dots 0\rangle\langle 11\dots 1| + \frac{1}{2}|11\dots 1\rangle\langle 00\dots 0|$$
* $Z$ measurements are completely blind to these off-diagonal terms.

### Trap 2: Only measuring the global $X$ parity
* The global $X$ observable $\Pi_X = X_0 X_1 \dots X_{n-1}$ flips all bits:
  $$\Pi_X (|0\dots 0\rangle + |1\dots 1\rangle) = |1\dots 1\rangle + |0\dots 0\rangle$$
  so $\langle GHZ_n | \Pi_X | GHZ_n \rangle = +1$, whereas $\operatorname{Tr}(\rho_{\text{mix}} \Pi_X) = 0$.
* But consider the unentangled product state:
  $$|+\rangle^{\otimes n} = \left(\frac{|0\rangle + |1\rangle}{\sqrt{2}}\right)^{\otimes n}$$
  Since $X|+\rangle = +|+\rangle$, we have $\langle +|^{\otimes n} \Pi_X |+\rangle^{\otimes n} = +1$.
  So an $X$ parity alone cannot distinguish $|GHZ_n\rangle$ from an unentangled product state!

### The Solution: Combined Parity Checks
* The $n-1$ adjacent $ZZ$ parities $Z_i Z_{i+1}$ ensure the state is strictly supported on equal neighboring bits (all-zeros or all-ones).
* The single global $X$ parity $\Pi_X$ guarantees that these two components are in a coherent quantum superposition with phase $+1$.
* Together, these $n$ operators form the **stabilizer generators** of the GHZ state!

---

## 3. Circuit Preparation
Starting from $|0\dots 0\rangle$:
1. Apply $H$ on qubit $0$:
   $$|0\dots 0\rangle \xrightarrow{H_0} \frac{|0\rangle + |1\rangle}{\sqrt{2}} |0\dots 0\rangle$$
2. Apply CNOTs with qubit $0$ as control and each wire $w \in \{1, \dots, n-1\}$ as target:
   $$\xrightarrow{CNOT(0, w)} \frac{1}{\sqrt{2}} (|00\dots 0\rangle + |11\dots 1\rangle)$$

---

## 4. Implementation in PennyLane
```python
def prepare_and_build_checks(n):
    # Step 1: Prepare GHZ state
    qp.Hadamard(wires=0)
    for wire in range(1, n):
        qp.CNOT(wires=[0, wire])

    # Step 2: Global X parity
    checks = [qp.prod(*(qp.PauliX(wires=w) for w in range(n)))]

    # Step 3: Adjacent ZZ parities
    for i in range(n - 1):
        checks.append(qp.prod(qp.PauliZ(wires=i), qp.PauliZ(wires=i + 1)))

    # Step 4: Return all n observables
    return checks
```
