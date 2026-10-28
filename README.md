# QHack 2026 Solutions

[![QHack 2026](https://img.shields.io/badge/QHack-2026-blueviolet?style=for-the-badge)](https://pennylane.ai/qhack)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.45.1-teal?style=for-the-badge)](https://pennylane.ai/)
[![Completion Status](https://img.shields.io/badge/Challenges-22%2F23%20Completed-brightgreen?style=for-the-badge)](#progress-overview)

This repository contains full implementations, mathematical breakdowns, and documentation for the challenges across all three groups in the QHack 2026 competition.

---

## Competition Profile

- **PennyLane / QHack Profile**: [The_Judge26](https://pennylane.ai/profile/The_Judge26)

---

## Progress Overview

| Challenge Group | Completed | Total | Status |
| :--- | :---: | :---: | :--- |
| **Group 1** | **7** | 7 | 100% Completed (7/7) |
| **Group 2** | **6** | 7 | 85.7% Completed (6/7) (1 unresolved challenge) |
| **Group 3** | **9** | 9 | 100% Completed (9/9) |
| **Total** | **22** | **23** | **95.7% Overall Completion** |

> [!NOTE]
> All challenges across **Group 1**, **Group 2**, and **Group 3** have been fully solved and verified, with the sole exception of **"Approximation Ratio for QAOA"** in Group 2. 
> 
> A comprehensive post-mortem and analysis of why this problem's reference solution is mathematically stalled (discrepancy between the expected partially-converged local minimum and true global convergence) is documented in [Group_2/approximation_ratio_for_qaoa/README.md](Group_2/approximation_ratio_for_qaoa/README.md).

---

## Challenges Directory

### Group 1 (7 / 7 Completed)
| Challenge | Folder | Description / Topic |
| :--- | :--- | :--- |
| **Diffuse and Conquer** | [`Group_1/diffuse_and_conquer`](Group_1/diffuse_and_conquer) | Grover diffusion operators and amplitude amplification |
| **Fermionic VQE** | [`Group_1/fermionic_vqe`](Group_1/fermionic_vqe) | Variational Quantum Eigensolver for fermionic systems |
| **Goal or No Goal** | [`Group_1/goal_or_no_goal`](Group_1/goal_or_no_goal) | State discrimination and hypothesis testing |
| **Grover's Little Helper** | [`Group_1/grovers_little_helper`](Group_1/grovers_little_helper) | Quantum search oracle construction |
| **Playing Period** | [`Group_1/playing_period`](Group_1/playing_period) | Quantum Phase Estimation and period finding |
| **The Draft Lottery** | [`Group_1/the_draft_lottery`](Group_1/the_draft_lottery) | Quantum random sampling and permutation circuits |
| **The Hidden Subgroup Stage**| [`Group_1/the_hidden_subgroup_stage`](Group_1/the_hidden_subgroup_stage) | Hidden Subgroup Problem (HSP) over finite groups |

---

### Group 2 (6 / 7 Completed)
| Challenge | Folder | Status | Description / Topic |
| :--- | :--- | :---: | :--- |
| **A Basket of Options** | [`Group_2/a_basket_of_options`](Group_2/a_basket_of_options) | Done | Quantum amplitude estimation for option pricing |
| **A Relay of Phases** | [`Group_2/a_relay_of_phases`](Group_2/a_relay_of_phases) | Done | Controlled phase shifts and multi-qubit gates |
| **Building a Quantum Kernel**| [`Group_2/building_a_quantum_kernel`](Group_2/building_a_quantum_kernel) | Done | Quantum Kernel methods for support vector machines |
| **Keep Your Own Score** | [`Group_2/keep_your_own_score`](Group_2/keep_your_own_score) | Done | Variational expectation value scoring and fidelity |
| **Nuclear Magnetic Resonance**| [`Group_2/nuclear_magnetic_resonance`](Group_2/nuclear_magnetic_resonance) | Done | Spin dynamics simulation and NMR pulse sequences |
| **The Scouting Report** | [`Group_2/the_scouting_report`](Group_2/the_scouting_report) | Done | Quantum feature mapping and dataset classification |
| **Approximation Ratio for QAOA**| [`Group_2/approximation_ratio_for_qaoa`](Group_2/approximation_ratio_for_qaoa) | Unresolved | Depth-p=4 QAOA for Minimum Vertex Cover ([Post-Mortem](Group_2/approximation_ratio_for_qaoa/README.md)) |

---

### Group 3 (9 / 9 Completed)
| Challenge | Folder | Description / Topic |
| :--- | :--- | :--- |
| **Break the Tie** | [`Group_3/break_the_tie`](Group_3/break_the_tie) | Parity checking and tie-breaking quantum measurements |
| **Implementing POVMs** | [`Group_3/implementing_povms`](Group_3/implementing_povms) | Positive Operator-Valued Measures via Naimark dilation |
| **My First Quantum Neural Network**| [`Group_3/my_first_quantum_neural_network`](Group_3/my_first_quantum_neural_network) | Parameterized QNN classification with Pennylane |
| **Personal Best (Bronze)** | [`Group_3/personal_best_bronze`](Group_3/personal_best_bronze) | Quantum circuit optimization (Bronze Tier) |
| **Personal Best (Silver)** | [`Group_3/personal_best_silver`](Group_3/personal_best_silver) | Quantum circuit optimization (Silver Tier) |
| **Personal Best (Gold)** | [`Group_3/personal_best_gold`](Group_3/personal_best_gold) | Quantum circuit optimization (Gold Tier) |
| **Pulse Optimization** | [`Group_3/pulse_optimization`](Group_3/pulse_optimization) | Hamiltonian pulse-level control and shape parameterization |
| **Same Kit Different Number**| [`Group_3/same_kit_different_number`](Group_3/same_kit_different_number) | Qubit permutation equivalence and Clifford group analysis |
| **The Salary Cap** | [`Group_3/the_salary_cap`](Group_3/the_salary_cap) | Constrained combinatorial optimization on graphs |

---

## Environment & Setup

To run and verify the solutions locally:

1. **Activate the virtual environment**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
2. **Install core requirements**:
   ```powershell
   pip install pennylane scipy networkx rustworkx
   ```
3. **Execute any solution**:
   ```powershell
   python Group_1/diffuse_and_conquer/diffuse_and_conquer.py
   ```
