# Challenge: The Hidden Subgroup Stage (200 pts)

## 1. Problem Overview
The **Hidden Subgroup Problem (HSP)** over $\mathbb{F}_2^n$ seeks to detect whether a dataset's class labels are invariant under a symmetry $s \in \mathbb{F}_2^n \setminus \{0\}$, meaning:
$$f(x) = f(x \oplus s)$$

For each class $r$ with samples $X_r = \{x_i : f(x_i) = r\}$:
1. Encode into uniform superposition $|X_r\rangle = \frac{1}{\sqrt{|X_r|}} \sum_{x \in X_r} |x\rangle$.
2. Apply Hadamard transformation $H^{\otimes n}$.
3. Use an ancilla qubit initialized to $|0\rangle$, applying $\text{CNOT}(j, \text{ancilla})$ for each wire $j$ where $s_j = 1$.
   This computes the parity $y \cdot s \pmod 2$ onto the ancilla.
4. The probability of finding the ancilla in $|0\rangle$ is:
   $$p_r(s) = P(y \cdot s = 0)$$
5. The overall symmetry score is the weighted average across classes:
   $$S(s) = \sum_r \frac{|X_r|}{N} p_r(s)$$

---

## 2. Mathematical Derivation & Analytical Solution

The state after the Hadamards is:
$$|\psi\rangle = H^{\otimes n} |X_r\rangle = \frac{1}{\sqrt{|X_r|} 2^{n/2}} \sum_{y \in \mathbb{F}_2^n} \left(\sum_{x \in X_r} (-1)^{x \cdot y}\right) |y\rangle$$

The probability of measuring ancilla $|0\rangle$ (meaning $y \cdot s = 0$) is:
$$p_r(s) = \sum_{y : y \cdot s = 0} |\langle y | \psi \rangle|^2 = \frac{1}{|X_r| 2^n} \sum_{x, x' \in X_r} \sum_{y : y \cdot s = 0} (-1)^{(x \oplus x') \cdot y}$$

By character orthogonality on the subgroup $s^\perp = \{y : y \cdot s = 0\}$:
$$\sum_{y : y \cdot s = 0} (-1)^{z \cdot y} = \begin{cases} 2^{n-1} & \text{if } z \in \{0, s\} \\ 0 & \text{otherwise} \end{cases}$$

Therefore:
$$p_r(s) = \frac{2^{n-1}}{|X_r| 2^n} \sum_{x, x' \in X_r} \mathbb{I}(x \oplus x' \in \{0, s\}) = \frac{1}{2 |X_r|} (|X_r| + 2 M_r) = \frac{1}{2} + \frac{M_r}{|X_r|}$$
where $M_r$ is the number of complete symmetric pairs $\{x, x \oplus s\} \subseteq X_r$.

Substituting this into the total symmetry score $S(s)$:
$$S(s) = \sum_r \frac{|X_r|}{N} \left(\frac{1}{2} + \frac{M_r}{|X_r|}\right) = \frac{1}{2}\sum_r \frac{|X_r|}{N} + \frac{1}{N}\sum_r M_r = \frac{1}{2} + \frac{M_{\text{total}}}{N}$$
where $M_{\text{total}}$ is the total number of symmetric pairs in the dataset that share the same class label!

---

## 3. Implementation
The solution supports both the exact quantum circuit (for $n \le 10$) and the analytical Fourier formula (which executes in $O(N)$ time, preventing memory exhaustion on hidden test cases with many qubits).
