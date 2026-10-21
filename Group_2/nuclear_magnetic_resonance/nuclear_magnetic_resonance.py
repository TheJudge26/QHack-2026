import json
try:
    import jax
    import jax.numpy as jnp
    has_jax = True
except ImportError:
    import numpy as jnp
    has_jax = False

import pennylane as qp
import numpy as np
from scipy.integrate import solve_ivp

dev = qp.device('default.qubit', wires = 1)

@qp.qnode(dev)
def nmr_evolve_circuit(B0: float, B1: float, omega: float, time: float):
    f_x = lambda p, t: p[0] * jnp.cos(p[1] * t)
    f_y = lambda p, t: -p[0] * jnp.sin(p[1] * t)

    H = B0 * qp.PauliZ(0) + f_x * qp.PauliX(0) + f_y * qp.PauliY(0)
    qp.evolve(H)([[B1, omega], [B1, omega]], t=time)

    return qp.probs(wires=0)


def nmr_evolve(B0: float, B1: float, omega: float, time: float):
    """This quantum circuit evolves the initial state |0> under the evolution operator generated
    by the nuclear magnetic resonance Hamiltonian.
    
    Args:
        B0 (float): Coefficient of the Pauli Z term in the Hamiltonian.
        B1 (float): Coefficient of the time-dependent component of the Hamiltonian.
        omega (float): Magnetic field oscillation frequency.
        time (float): Period of time during which the magnetic field acts on the qubit.

    Returns:
        (numpy.array): The probabilities of measuring each computational basis state.
    """
    if has_jax:
        return nmr_evolve_circuit(B0, B1, omega, time)
    else:
        # Exact numerical ODE solution matching Schrödinger equation
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        def ode(t, y):
            psi = y[:2] + 1j * y[2:]
            H = B0 * Z + B1 * (np.cos(omega * t) * X - np.sin(omega * t) * Y)
            dpsi = -1j * (H @ psi)
            return np.concatenate([dpsi.real, dpsi.imag])

        y0 = np.array([1.0, 0.0, 0.0, 0.0])
        sol = solve_ivp(ode, [0, time], y0, rtol=1e-10, atol=1e-10)
        y_final = sol.y[:, -1]
        psi_final = y_final[:2] + 1j * y_final[2:]
        return np.abs(psi_final)**2


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    ins = json.loads(test_case_input)
    output = nmr_evolve(*ins).tolist()

    return str(output)

def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(
        np.array(solution_output), np.array(expected_output), atol=0.03
    ), "Your code does not yield the correct probabilities"


# These are the public test cases
test_cases = [
    ('[1,2,0.5,0.3]', '[0.6962061524391174, 0.3037952184677124]'),
    ('[1, 1, 0.25, 0.6]', '[0.7278291583061218, 0.272170752286911]')
]

if __name__ == "__main__":
    for i, (input_, expected_output) in enumerate(test_cases):
        print(f"Running test case {i} with input '{input_}'...")

        try:
            output = run(input_)

        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"Runtime Error. {exc}")

        else:
            if message := check(output, expected_output):
                print(f"Wrong Answer. Have: '{output}'. Want: '{expected_output}'.")

            else:
                print("Correct!")
