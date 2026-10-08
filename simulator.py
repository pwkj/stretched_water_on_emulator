#https://github.com/HQC2/SQ_tutorials
import numpy as np
import numba as nb
import matplotlib.pyplot as plt

import pyscf
from pyscf import scf, mcscf

import slowquant.SlowQuant as sq
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS
from slowquant.qiskit_interface.interface import QuantumInterface
from slowquant.qiskit_interface.circuit_wavefunction import WaveFunctionCircuit

from qiskit_aer.primitives import SamplerV2
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.circuit import QuantumCircuit, Gate


def count_qubit_gates(circuit):
    # Step 1: Decompose as much as possible
    decomposed = circuit.decompose(reps=10)

    # Step 2: Count gates
    one_qubit_gate_count = 0
    two_qubit_gate_count = 0
    for instr, qargs, cargs in decomposed.data:
        if len(qargs) == 1:
            one_qubit_gate_count += 1
        if len(qargs) == 2:
            two_qubit_gate_count += 1


    return one_qubit_gate_count, two_qubit_gate_count


def ideal_simulator(mol, wf, active_space, n_layers, runs):

    print("4 threads")
    nb.set_num_threads(4)

    # PySCF 
    rhf = pyscf.scf.RHF(mol).run()
    mo_coeff = rhf.mo_coeff
    integral_generator = mol   

    print("### Number of orbitals ###", len(mo_coeff))

    energies = []   
    thetas = []
    
    WF_wo_oo = WaveFunctionUPS(
    active_space, # active space (num_elec, num_orbs)
    mo_coeff,
    integral_generator,
    wf, # Ansatz
    ansatz_options={"n_layers": n_layers, "skip_last_singles": True}, # NO skip_last_singles option as we do not use oo
    include_active_kappa=False, # No oo. Default is false, so can also be removed
    )

    WF_wo_oo.run_wf_optimization_1step("BFGS", orbital_optimization = False)
    #WF_wo_oo.run_wf_optimization_2step("rotosolve", orbital_optimization = True, tol = 1e-3, maxiter = 200)
    
    
    WF = WaveFunctionUPS(
    active_space, # active space (num_elec, num_orbs)
    mo_coeff,
    integral_generator,
    wf, # Ansatz
    ansatz_options={"n_layers": n_layers, "skip_last_singles": True}, # Options
    include_active_kappa=True,
    )
    
    #WF.thetas =  WF_wo_oo.thetas
    WF.run_wf_optimization_1step("BFGS", orbital_optimization = True, tol = 1e-8, maxiter = 500)
    #WF.run_wf_optimization_2step("rotosolve", orbital_optimization = True, tol = 1e-8, maxiter = 200)
    energies.append(float(WF.energy_elec + mol.energy_nuc()))
    thetas.append(WF.thetas)

    print("### Energy ###",energies[0])

    if runs > 0:
        for i in range(runs):
            WF = WaveFunctionUPS(
            active_space, # active space (num_elec, num_orbs)
            mo_coeff,
            integral_generator,
            wf, # Ansatz
            ansatz_options={"n_layers": n_layers, "skip_last_singles": True},
            include_active_kappa=True,
            )

            WF.thetas = (np.random.random(len(WF.thetas))*2*np.pi).tolist()
            WF.run_wf_optimization_1step("BFGS", orbital_optimization = True)
            #WF.run_wf_optimization_2step("rotosolve", orbital_optimization = True, tol = 1e-8, maxiter = 200)
            energies.append(float(WF.energy_elec + mol.energy_nuc()))
            thetas.append(WF.thetas)

            print("### Energy ###",energies[i+1]) 

    min_idx = np.argmin(energies)
    min_energy = energies[min_idx]
    min_thetas = thetas[min_idx]

    return energies, thetas, min_energy, min_thetas



def shot_noise_simulator(mol, wf, n_layers, active_space, shots, runs):

    # PySCF 
    rhf = pyscf.scf.RHF(mol).run()
    mo_coeff = rhf.mo_coeff
    integral_generator = mol

   # Define the sampler from QiskitAer that will be used for the quantum emulation
    sampler = SamplerV2()

    # Define the mapper that will be used to translate fermionic operators to Pauli strings in the qubit representation
    mapper = JordanWignerMapper()

    ansatz_options={"n_layers": n_layers, "skip_last_singles": True} # Options
    
    QI = QuantumInterface(
    sampler, # pass sampler
    wf, # Ansatz
    mapper, # pass mapper
    pass_manager_options = {"optimization_level": 3},
    ansatz_options=ansatz_options,
    shots=shots,
    )

    QI._save_paulis = True  # hard switch to stop using Pauli saving (debugging tool).
    QI._do_cliques = True # hard switch to stop using QWC (debugging tool).
    
    print("### QI info ###")
    QI.get_info()

    energies = []
    opt_angles = []
    opt_c_mo = []
    total_shots_used = []
    total_paulis_evaluated = []

    qWF = WaveFunctionCircuit(
    active_space,
    mo_coeff,  
    integral_generator,
    QI,  # pass QuantumInterface
    include_active_kappa = True  
    )

    print("### Total one-qubit gates ###", count_qubit_gates(circuit=qWF.QI.circuit)[0]+active_space[0])
    print("### Total two-qubit gates ###", count_qubit_gates(circuit=qWF.QI.circuit)[1])
    print("### Circuit depth ###", qWF.QI.circuit.depth())
    print("### Circuit width (#qubits) ###", qWF.QI.circuit.width())

    large_font = {"fontsize": 20,"subfontsize": 8}
    print(qWF.QI.circuit.decompose().decompose().decompose().draw(output="mpl", fold=50, style=large_font))
    plt.savefig('non_transpiled_circuit.png')

    # Run wave function optimization
    # We have adjusted the tolerance and maxiter to looser values because it will be hard to converge shot noise
    #qWF.run_wf_optimization_2step("rotosolve", orbital_optimization = True, tol = 1e-3, maxiter = 200)
    qWF.run_wf_optimization_2step("BFGS", orbital_optimization = True, tol = 1e-3, maxiter = 500)

    opt_angles.append(qWF.thetas)
    opt_c_mo.append(qWF.c_mo)
    total_shots_used.append(qWF.QI.total_shots_used)
    total_paulis_evaluated.append(qWF.QI.total_paulis_evaluated)

    energies.append(qWF.energy_elec + mol.energy_nuc())  

    print("### Total Paulis evaluated ###",total_paulis_evaluated[0])
    print("### Energy ###",energies[0])  

    # The QuantumInterface saves the results from the quantum emulation run. In order to run a new quanutm emulation, you have to reset the QI:
    QI._reset_cliques()

    if runs > 0:
        for i in range(runs):
            qWF = WaveFunctionCircuit(
            active_space,
            mo_coeff, 
            integral_generator,
            QI,  # pass QuantumInterface
            include_active_kappa = True  
            )
            #qWF.thetas = (np.random.random(len(qWF.thetas))*2*np.pi).tolist()
            qWF.thetas = opt_angles[i] # Pass the theta parameters from ideal simulator to circuit wave function
            qWF.run_wf_optimization_2step("rotosolve", orbital_optimization = True, tol = 1e-3, maxiter = 200)

            opt_angles.append(qWF.thetas)
            opt_c_mo.append(qWF.c_mo)
            total_shots_used.append(qWF.QI.total_shots_used)
            total_paulis_evaluated.append(qWF.QI.total_paulis_evaluated)

            energies.append(qWF.energy_elec + mol.energy_nuc())

            print("### Total Paulis evaluated ###",qWF.QI.total_paulis_evaluated)
            print("### Energy ###",energies[i+1])  

            # The QuantumInterface saves the results from the quantum emulation run. In order to run a new quanutm emulation, you have to reset the QI:
            QI._reset_cliques()

    return energies, total_shots_used, total_paulis_evaluated




