# Nuclear Magnetic Resonance (200 pts)

## Problem Overview
Nuclear Magnetic Resonance (NMR) describes the quantum dynamics of a magnetic spin (qubit) subjected to a static magnetic field along the $\hat{z}$-axis and a perpendicular, rotating magnetic field in the $xy$-plane.

The Hamiltonian is time-dependent:
$$H(t) = B_0 Z + B_1 (\cos(\omega t) X - \sin(\omega t) Y)$$

where:
- $B_0$: Static longitudinal magnetic field strength along $Z$.
- $B_1$: Amplitude of the rotating transversal field in the $xy$-plane.
- $\omega$: Angular frequency of the rotating field.

The system begins in the ground state $|\psi(0)\rangle = |0\rangle$ and evolves for duration $t = \text{time}$ under the time-ordered evolution operator:
$$U(t) = \mathcal{T} \exp\left(-i \int_0^t H(\tau) \, d\tau\right)$$

The goal is to compute the computational basis probabilities $P(0)$ and $P(1)$ of the state at time $t$.

---

## 1. Physics & The Rotating Frame Transformation
By applying a unitary rotation into the frame rotating at frequency $\omega$:
$$R(t) = \exp\left(i \frac{\omega t}{2} Z\right)$$
the state in the rotating frame is $|\psi_R(t)\rangle = R(t)|\psi(t)\rangle$.

In this frame:
$$R(t) (\cos(\omega t) X - \sin(\omega t) Y) R^\dagger(t) \approx X$$
The effective Hamiltonian in the rotating frame becomes predominantly static:
$$H_{\text{eff}} \approx \left(B_0 - \frac{\omega}{2}\right) Z + B_1 X$$

This yields Rabi oscillations between $|0\rangle$ and $|1\rangle$ at the generalized Rabi frequency:
$$\Omega = \sqrt{\left(B_0 - \frac{\omega}{2}\right)^2 + B_1^2}$$

PennyLane's pulse simulator computes the exact full time-ordered evolution without requiring the rotating-wave approximation by numerically integrating the Schrödinger equation via JAX ODE solvers.

---

## 2. PennyLane Pulse Implementation
In PennyLane's pulse module:
1. Construct the parameterized time-dependent functions using `jax.numpy` (`jnp`):
   ```python
   f_x = lambda p, t: p[0] * jnp.cos(p[1] * t)
   f_y = lambda p, t: -p[0] * jnp.sin(p[1] * t)
   ```
2. Combine the drift term $B_0 Z$ and the time-dependent terms into a `ParametrizedHamiltonian`:
   ```python
   H = B0 * qp.PauliZ(0) + f_x * qp.PauliX(0) + f_y * qp.PauliY(0)
   ```
3. Evolve using `qp.evolve(H)(params, t=time)`:
   ```python
   qp.evolve(H)([[B1, omega], [B1, omega]], t=time)
   ```
4. Return the measurement:
   ```python
   return qp.probs(wires=0)
   ```

---

## 3. Code Solution
```python
@qp.qnode(dev)
def nmr_evolve(B0: float, B1: float, omega: float, time: float) -> "jnp.ndarray":
    f_x = lambda p, t: p[0] * jnp.cos(p[1] * t)
    f_y = lambda p, t: -p[0] * jnp.sin(p[1] * t)

    H = B0 * qp.PauliZ(0) + f_x * qp.PauliX(0) + f_y * qp.PauliY(0)
    qp.evolve(H)([[B1, omega], [B1, omega]], t=time)

    return qp.probs(wires=0)
```
