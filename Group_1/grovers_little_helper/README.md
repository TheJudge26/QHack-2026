# Challenge: Grover's Little Helper (400 pts)

## 1. Problem Overview
In standard Grover's search on $N$ items, when the number of solutions is half the total items ($M = N/2$):
* The mean amplitude across all states after the oracle is:
  $$\mu = \frac{1}{N} \left( \sum_{x \notin S} 1 + \sum_{x \in S} (-1) \right) = \frac{(N/2) - (N/2)}{N} = 0$$
* The Grover diffusion operator reflects about the mean:
  $$2\mu - a_x = -a_x$$
* All probabilities remain completely unchanged ($|a_x|^2 = 1/N$), so amplitude amplification fails!

---

## 2. Overcoming the Barrier: Search Space Doubling with 1 Auxiliary Qubit

To solve this, we introduce **1 auxiliary qubit** (wire 2), expanding the state space from $N = 4$ to $N' = 8$:
1. **Initial Superposition**: Prepare the 3-qubit uniform state:
   $$|\psi_3\rangle = H^{\otimes 3} |000\rangle = |+++\rangle$$
2. **Controlled Oracle**:
   Condition the oracle on the auxiliary qubit being $|0\rangle$:
   $$\text{ctrl}(\text{oracle}, \text{control}=2, \text{control\_values}=[0])$$
   * Out of the 8 states, exactly $M = 2$ states are marked ($|s_1, 0\rangle$ and $|s_2, 0\rangle$).
   * The fraction of marked states is now:
     $$\frac{M}{N'} = \frac{2}{8} = \frac{1}{4}$$
3. **Exact 1-Shot Convergence**:
   In Grover's algorithm, the angle $\theta$ is given by:
   $$\sin(\theta) = \sqrt{\frac{M}{N'}} = \sqrt{\frac{1}{4}} = \frac{1}{2} \implies \theta = 30^\circ$$
   * A single Grover iteration rotates the state by $2\theta = 60^\circ$:
     $$\theta + 2\theta = 3\theta = 90^\circ$$
   * Because $\sin(90^\circ) = 1.0$, the quantum state rotates **with exactly 100% fidelity** into the marked subspace!
   * Therefore, in **a single shot**, measuring wires 0 and 1 yields one of the true solutions with probability 1.

---

## 3. Implementation
```python
def circuit(generator, oracle, diffuser):
    @qp.prod
    def gen3():
        generator()
        qp.Hadamard(wires=2)

    gen3()
    qp.ctrl(oracle, control=2, control_values=[0])()
    diffuser(gen3)

    return qp.counts(wires=[0, 1])
```
