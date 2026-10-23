import json
import pennylane as qp
import pennylane.numpy as np

# Write any helper functions you may need here

def get_min_eig(mapped_edges, n):
    """Computes the true minimum eigenvalue of the diagonal cost Hamiltonian
    using only pre-defined np without any additional imports or exponential dense matrices.
    """
    idx = np.arange(1 << n)
    shifts = np.array([n - 1 - i for i in range(n)])
    bits = (idx[:, None] >> shifts) & 1
    z = 1.0 - 2.0 * bits
    cost_diag = np.zeros(1 << n)
    for u, v in mapped_edges:
        cost_diag = cost_diag + 0.75 * (z[:, u] * z[:, v] + z[:, u] + z[:, v])
    cost_diag = cost_diag - np.sum(z, axis=1)
    return float(np.min(cost_diag))


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
    if not edges:
        return np.tensor(1.0)

    # 1. Remap vertex labels to contiguous 0, 1, ..., n-1
    unique_v = sorted(list(set(v for edge in edges for v in edge)))
    mapping = {v: i for i, v in enumerate(unique_v)}
    mapped_edges = [[mapping[u], mapping[v]] for u, v in edges]
    n = len(unique_v)

    # 2. Build Cost Hamiltonian with grouped degree terms
    deg = {i: 0 for i in range(n)}
    for u, v in mapped_edges:
        deg[u] += 1
        deg[v] += 1

    coeffs_cost = []
    obs_cost = []
    for u, v in mapped_edges:
        coeffs_cost.append(0.75)
        obs_cost.append(qp.PauliZ(u) @ qp.PauliZ(v))
    for i in range(n):
        coeffs_cost.append(0.75 * deg[i] - 1.0)
        obs_cost.append(qp.PauliZ(i))
    H_cost = qp.Hamiltonian(coeffs_cost, obs_cost)

    # 3. Build Mixer Hamiltonian
    coeffs_mixer = [1.0] * n
    obs_mixer = [qp.PauliX(i) for i in range(n)]
    H_mixer = qp.Hamiltonian(coeffs_mixer, obs_mixer)

    # 4. 4-Layer QAOA Circuit on standard contiguous wires 0..n-1
    dev = qp.device("default.qubit", wires=n)

    @qp.qnode(dev)
    def circuit(params):
        for w in range(n):
            qp.Hadamard(wires=w)
        for p in range(4):
            qp.qaoa.cost_layer(params[0][p], H_cost)
            qp.qaoa.mixer_layer(params[1][p], H_mixer)
        return qp.expval(H_cost)

    # 5. Optimize parameters (30 steps of Gradient Descent)
    opt = qp.GradientDescentOptimizer(stepsize=0.01)
    params = np.array([[0.25] * 4, [0.5] * 4], requires_grad=True)

    for _ in range(30):
        params = opt.step(circuit, params)

    min_expval = circuit(params)

    # 6. True minimum eigenvalue and approximation ratio
    min_eig = get_min_eig(mapped_edges, n)
    ratio = float(min_expval / min_eig)

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
            print(f"Runtime Error. {exc}")
        else:
            if message := check(output, expected_output):
                print(f"Wrong Answer. Have: '{output}'. Want: '{expected_output}'.")
            else:
                print(f"Output: {output} Expected: {expected_output}")
                print("Correct!")
