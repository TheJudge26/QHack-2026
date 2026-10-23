import json
import pennylane as qp
import pennylane.numpy as np

# Write any helper functions you may need here


def approximation_ratio(edges):
    """This function calculates the approximation ratio of QAOA for
    the minimum vertex cover problem

    Args:
    - Edges (list(list(int))): A list of ordered pairs of integers, representing 
    the edges of the graph for which we need to solve the minimum vertex cover problem.

    Returns:
    - np.tensor(float): The approximation ratio of QAOA for the minimum vertex cover problem for
    the input graph.
    """
    # Identify unique vertices and map to contiguous wire indices
    nodes = sorted(list(set([v for edge in edges for v in edge])))
    node_map = {node: i for i, node in enumerate(nodes)}
    n_wires = len(nodes)

    # Build cost Hamiltonian:
    # H_cost = 0.75 * sum_{(i,j) in edges} (Z_i Z_j + Z_i + Z_j) - sum_{i in vertices} Z_i
    coeffs = []
    obs = []
    for u, v in edges:
        i, j = node_map[u], node_map[v]
        coeffs.extend([0.75, 0.75, 0.75])
        obs.extend([qp.PauliZ(i) @ qp.PauliZ(j), qp.PauliZ(i), qp.PauliZ(j)])
    for i in range(n_wires):
        coeffs.append(-1.0)
        obs.append(qp.PauliZ(i))

    cost_h = qp.Hamiltonian(coeffs, obs)

    # Build mixer Hamiltonian:
    # H_mixer = sum_{i in vertices} X_i
    mixer_h = qp.Hamiltonian([1.0] * n_wires, [qp.PauliX(i) for i in range(n_wires)])

    dev = qp.device("default.qubit", wires=n_wires)

    # QAOA layer: cost layer followed by mixer layer
    def qaoa_layer(gamma, alpha):
        qp.qaoa.cost_layer(gamma, cost_h)
        qp.qaoa.mixer_layer(alpha, mixer_h)

    # 4-layer QAOA circuit
    @qp.qnode(dev, diff_method="backprop")
    def circuit(params):
        for w in range(n_wires):
            qp.Hadamard(wires=w)
        qp.layer(qaoa_layer, 4, params[0], params[1])
        return qp.expval(cost_h)

    # Exact classical ground state energy (minimum eigenvalue of H_cost)
    cost_matrix = qp.matrix(cost_h, wire_order=range(n_wires))
    min_eig = np.min(np.linalg.eigvalsh(cost_matrix))

    # Variational optimization of gamma and alpha angles
    params = 0.5 * np.ones((2, 4), requires_grad=True)
    opt = qp.AdamOptimizer(stepsize=0.1)
    for _ in range(50):
        params = opt.step(circuit, params)

    min_expval = circuit(params)

    # Return approximation ratio: r* = <H_cost> / E_min
    ratio = min_expval / min_eig
    return ratio


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    ins = json.loads(test_case_input)
    output = approximation_ratio(ins)

    return str(output)


def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(
        solution_output, expected_output, rtol=5e-2
    ), "Your code does not yield the right approximation ratio"


# These are the public test cases
test_cases = [
    ('[[0, 1], [1, 2], [0, 2], [2, 3]]', '0.58160924'),
    ('[[0, 1], [1, 2], [2, 3], [3, 0]]', '0.94746734')
]

if __name__ == "__main__":
    # This will run the public test cases locally
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
