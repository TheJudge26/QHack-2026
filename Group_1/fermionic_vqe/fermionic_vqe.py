import json
import pennylane as qp
import pennylane.numpy as np

# Define the Pauli words
w0 = qp.jordan_wigner(qp.fermi.from_string('0 0+'), ps=True)
w1 = qp.jordan_wigner(qp.fermi.from_string('1 1+'), ps=True)
w2 = qp.jordan_wigner(qp.fermi.from_string('2 2+'), ps=True)


def build_hamiltonian(coeffs):
    """
    Build the Hamiltonian with the input coefficients.
    Args:
        coeffs (list(float)): A list of coefficients that will multiply the Pauli words to form the Pauli sentence that will be used to build the Hamiltonian.
    Returns: 
        An operator that corresponds with the desired Hamiltonian.
    """
    H_ps = coeffs[0] * w0 + coeffs[1] * w1 + coeffs[2] * w2
    H_ps_real = qp.pauli.PauliSentence({k: float(v.real) for k, v in H_ps.items()})
    return H_ps_real.operation()


dev = qp.device("default.qubit", wires=6)


@qp.qnode(dev)
def vqe_circuit(params, coeffs):
    """
    The quantum circuit of the VQE
    Args:
        params(np.array): An array with the trainable parameters of the Variational Quantum Eigensolver circuit.
        coeffs (list(float)): A list of coefficients that will multiply the Pauli words to form the Pauli sentence that will be used to build the Hamiltonian.
    Returns: 
        (np.tensor): A numpy tensor of 1 element corresponding to the expectation value of the cost Hamiltonian.
    """
    for i in range(6):
        qp.RY(params[i], wires=i)
    return qp.expval(build_hamiltonian(coeffs))


def optimization(coeffs):
    """
    A function to optimize the parameters of the VQE circuit.
    Args:
        coeffs (list(float)): A list of coefficients that will multiply the Pauli words to form the Pauli sentence that will be used to build the Hamiltonian.
    Returns: 
        (np.tensor): A numpy tensor of 1 element corresponding to the minimum eigenvalue of the Hamiltonian, i.e., the expectation value of the cost Hamiltonian
        with the parameters optimized.
    """
    opt = qp.GradientDescentOptimizer(stepsize=0.3)
    params = np.random.uniform(0.1, 2 * np.pi, 6, requires_grad=True)

    for _ in range(80):
        params = opt.step(lambda p: vqe_circuit(p, coeffs), params)

    return vqe_circuit(params, coeffs)


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    coeffs = json.loads(test_case_input)
    np.random.seed(40)
    output = optimization(coeffs)
    return str(output)


def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert abs(solution_output - expected_output) < 0.1


# These are the public test cases
test_cases = [
    ('[-2,2,5]', '-2'),
    ('[7,6,-6]', '-6')
]

if __name__ == "__main__":
    for i, (input_, expected_output) in enumerate(test_cases):
        print(f"Running test case {i} with input '{input_}'...")

        try:
            output = run(input_)
        except Exception as exc:
            print(f"Runtime Error. {exc}")
        else:
            if message := check(output, expected_output):
                print(f"Wrong Answer. Have: '{output}'. Want: '{expected_output}'.")
            else:
                print("Correct!")
