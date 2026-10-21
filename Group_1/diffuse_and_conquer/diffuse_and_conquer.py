import json
import pennylane as qp
import pennylane.numpy as np

# DO NOT CHANGE THESE VARIABLES LATER IN THE CODE

phi = np.array([0.03, 0.04, 0.07, 0.09])
source = np.array([1., 2., 1., 0.5])
dt = 0.01
h = 0.2

b = phi + dt * source

C = 0.28407950891345873

phi_qsvt = np.array([
    -3.9269606809117215, 1.5707899319523129, 1.5707992180163084, 1.570796656458965,
    1.5707936178641142, 1.570800320804173, 1.5707920871715215, 1.5708001396978588,
    1.5707926563375072, 1.5708030400613744, 1.5707738725775875, 1.5708766742642861,
    1.5705338807173796, 1.571568649749855, 1.568737117893249, 1.5758003758538,
    1.5596595828878401, 1.5935854401460452, 1.5277876343234589, 1.6457687077564584,
    1.4502785510214011, 1.7481342354176, 1.3355540553688812, 1.8460933148380323,
    1.8460933148380438, 1.3355540553688852, 1.7481342354175953, 1.4502785510214051,
    1.6457687077564576, 1.5277876343234595, 1.593585440146046, 1.5596595828878386,
    1.5758003758538002, 1.5687371178932485, 1.5715686497498562, 1.5705338807173808,
    1.5708766742642868, 1.5707738725775886, 1.5708030400613737, 1.5707926563375076,
    1.5708001396978577, 1.5707920871715215, 1.570800320804173, 1.5707936178641142,
    1.5707966564589646, 1.5707992180163082, 1.5707899319523122, -0.7853680273219278
])

D2 = (1.0 / (h ** 2)) * np.array([
    [-2,  1,  0,  0],
    [ 1, -2,  1,  0],
    [ 0,  1, -2,  1],
    [ 0,  0,  1, -2]
])

device = qp.device("default.qubit", wires=4)


@qp.qnode(device)
def qsvt_diffusion(nu, k):
    """
    Implements QSVT to find the concentration phi after one time step

    Args:
    - nu (float): The diffusion coefficient
    - k (float): The reaction rate

    Returns:
    - (np.array): The concentation vector on the next time step.
    """
    A = (1.0 + k * dt) * np.eye(4) - nu * dt * D2
    qp.StatePrep(b / np.linalg.norm(b), wires=[2, 3])

    be = qp.BlockEncode(A, wires=[1, 2, 3])
    projectors = [qp.PCPhase(angle, dim=4, wires=[1, 2, 3]) for angle in phi_qsvt]

    qp.Hadamard(wires=0)
    qp.ctrl(qp.QSVT, control=0)(be, projectors)
    qp.ctrl(qp.adjoint(qp.QSVT), control=0, control_values=[0])(be, projectors)
    qp.Hadamard(wires=0)

    return qp.state()


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    ins = json.loads(test_case_input)
    out_state = qsvt_diffusion(*ins)[:4]
    rescaled_out = out_state * np.linalg.norm(b) / C
    output = rescaled_out / np.linalg.norm(rescaled_out)

    return str(np.real(output).tolist())


def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    fidelity = np.dot(solution_output, expected_output)
    assert fidelity >= 0.99, "Your circuit doesn't look quite right "


# These are the public test cases
test_cases = [
    ('[0.2,0.5]', '[0.27959746, 0.42869397, 0.56892447, 0.64371709]'),
    ('[1.0, 1.2]', '[0.28550356, 0.4595721 , 0.59350887, 0.59584262]')
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
