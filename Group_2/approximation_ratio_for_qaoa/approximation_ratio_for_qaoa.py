import json
import pennylane as qp
import pennylane.numpy as np

# Write any helper functions you may need here
from scipy.optimize import minimize


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
    # Canonicalize unique undirected edges
    unique_edges = set()
    for u, v in edges:
        if u != v:
            unique_edges.add(tuple(sorted([u, v])))
    unique_edges = sorted(list(unique_edges))

    nodes = sorted(list(set([v for edge in unique_edges for v in edge])))
    n_wires = len(nodes)
    mapping = {v: i for i, v in enumerate(nodes)}

    # Precompute diagonal of cost Hamiltonian:
    # H_cost = 0.75 * sum_{(i,j) in edges} (Z_i Z_j + Z_i + Z_j) - sum_{i in vertices} Z_i
    diag_H = np.zeros(1 << n_wires)
    for idx in range(1 << n_wires):
        bits = [(1 - 2 * ((idx >> (n_wires - 1 - i)) & 1)) for i in range(n_wires)]
        val = 0.0
        for u, v in unique_edges:
            zi = bits[mapping[u]]
            zj = bits[mapping[v]]
            val += 0.75 * (zi * zj + zi + zj)
        for i in range(n_wires):
            val -= bits[i]
        diag_H[idx] = val

    # Classical ground state energy of H_cost
    min_eig = np.min(diag_H)

    # Ultra-fast exact statevector QAOA simulation for depth p = 4
    def cost_func(params):
        gamma = params[:4]
        alpha = params[4:]
        state = np.ones(1 << n_wires, dtype=complex) / np.sqrt(1 << n_wires)
        state = state.reshape([2] * n_wires)
        for p in range(4):
            # Cost layer: exp(-i * gamma_p * H_cost)
            state = (state.flatten() * np.exp(-1j * gamma[p] * diag_H)).reshape([2] * n_wires)
            # Mixer layer: exp(-i * alpha_p * X) applied to each wire
            c = np.cos(alpha[p])
            s = -1j * np.sin(alpha[p])
            rot = np.array([[c, s], [s, c]], dtype=complex)
            for q in range(n_wires):
                state = np.tensordot(rot, state, axes=(1, q))
                state = np.moveaxis(state, 0, q)
        probs = np.abs(state.flatten()) ** 2
        return float(np.sum(probs * diag_H))

    # Variational optimization from the standard initial point
    res = minimize(cost_func, 0.5 * np.ones(8), method='SLSQP')
    ratio = res.fun / min_eig
    return np.tensor(float(ratio))


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
                print("Correct!")
