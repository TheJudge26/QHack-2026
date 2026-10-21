import json
import pennylane as qp
import pennylane.numpy as np

# Write any helper functions you may need here

def build_hamiltonians(edges):
    vertices = sorted(list(set(v for edge in edges for v in edge)))

    cost_coeffs = []
    cost_terms = []
    for i, j in edges:
        cost_coeffs.extend([0.75, 0.75, 0.75])
        cost_terms.extend([qp.PauliZ(i) @ qp.PauliZ(j), qp.PauliZ(i), qp.PauliZ(j)])

    for i in vertices:
        cost_coeffs.append(-1.0)
        cost_terms.append(qp.PauliZ(i))

    H_cost = qp.Hamiltonian(cost_coeffs, cost_terms)

    mixer_coeffs = [1.0] * len(vertices)
    mixer_terms = [qp.PauliX(i) for i in vertices]
    H_mixer = qp.Hamiltonian(mixer_coeffs, mixer_terms)

    return H_cost, H_mixer, vertices


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
    H_cost, H_mixer, vertices = build_hamiltonians(edges)
    num_wires = len(vertices)
    dev = qp.device("default.qubit", wires=num_wires)

    depth = 4

    def qaoa_layer(gamma, beta):
        qp.qaoa.cost_layer(gamma, H_cost)
        qp.qaoa.mixer_layer(beta, H_mixer)

    @qp.qnode(dev)
    def circuit(params):
        for w in vertices:
            qp.Hadamard(wires=w)
        qp.layer(qaoa_layer, depth, params[0], params[1])
        return qp.expval(H_cost)

    opt = qp.AdamOptimizer(stepsize=0.1)
    params = np.array([[0.5] * depth, [0.5] * depth], requires_grad=True)

    for _ in range(50):
        params, cost = opt.step_and_cost(circuit, params)

    min_eig = np.min(qp.eigvals(H_cost))
    ratio = cost / min_eig

    return np.tensor(ratio)


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
