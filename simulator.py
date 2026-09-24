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


def ideal_simulator(mol, wf, active_space, n_layers, random_runs):

    print("4 threads")
    nb.set_num_threads(4)

    #fucc: 10.1021/acs.jctc.8b01004 (k-UpCCGSD)

    # PySCF 
    rhf = pyscf.scf.RHF(mol).run()
    mo_coeff = rhf.mo_coeff
    integral_generator = mol   
    energies = []

    print("Number of orbitals = ", len(mo_coeff))

    if wf == "tUPS":
        ansatz_options={"n_layers": n_layers, "skip_last_singles": True} # Options
    
    if wf == "fuccd":
        ansatz_options={"n_layers": n_layers, "D": True}
        wf = "fucc"
    
    if wf == "fuccpD":
        ansatz_options={"n_layers": n_layers, "pD": True}
        wf = "fucc"
    
    if wf == "fuccGpD":
        ansatz_options={"n_layers": n_layers, "GpD": True}
        wf = "fucc"

    WF_tUPS_wo_oo = WaveFunctionUPS(
    active_space, # active space (num_elec, num_orbs)
    mo_coeff,
    integral_generator,
    wf, # Ansatz
    ansatz_options=ansatz_options, 
    include_active_kappa=True,
    )

    WF_tUPS_wo_oo.run_wf_optimization_1step("BFGS", orbital_optimization = False)

    WF = WaveFunctionUPS(
    active_space, # active space (num_elec, num_orbs)
    mo_coeff,
    integral_generator,
    wf, # Ansatz
    ansatz_options=ansatz_options, # Options
    include_active_kappa=True,
    )
    WF.thetas =  WF_tUPS_wo_oo.thetas
    WF.run_wf_optimization_1step("BFGS", orbital_optimization = True)
    #WF.run_wf_optimization_2step("rotosolve", orbital_optimization = True, tol = 1e-3, maxiter = 200)
    energies.append(float(WF.energy_elec + mol.energy_nuc()))

    if random_runs > 0:

        for i in range(random_runs):

            WF = WaveFunctionUPS(
            active_space, # active space (num_elec, num_orbs)
            mo_coeff,
            integral_generator,
            wf, # Ansatz
            ansatz_options=ansatz_options, # Options
            include_active_kappa=True,
            )

            WF.thetas = (np.random.random(len(WF.thetas))*2*np.pi).tolist()
            WF.run_wf_optimization_1step("BFGS", orbital_optimization = True)
            #WF.run_wf_optimization_2step("rotosolve", orbital_optimization = True, tol = 1e-3, maxiter = 200)
            energies.append(float(WF.energy_elec + mol.energy_nuc()))

    print("best_energy:", min(energies))

    return min(energies), energies



def shot_noise(mol, active_space, shots, n_layers, random_runs, wf):

    # PySCF 
    rhf = pyscf.scf.RHF(mol).run()
    mo_coeff = rhf.mo_coeff
    integral_generator = mol

   # Define the sampler from QiskitAer that will be used for the quantum emulation
    sampler = SamplerV2()

    # Define the mapper that will be used to translate fermionic operators to Pauli strings in the qubit representation
    mapper = JordanWignerMapper()

    if wf == "tUPS":
        ansatz_options={"n_layers": n_layers, "skip_last_singles": True} # Options
    
    if wf == "fUCCD":
        ansatz_options={"n_layers": n_layers, "D": True}
        wf = "fUCC"
    
    if wf == "fUCCGpD":
        ansatz_options={"n_layers": n_layers, "GpD": True}
        wf = "fUCC"

    
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
    
    best_run = 0
    best_energy = 0
    opt_angles_tmp = []
    opt_c_mo_tmp = []
    total_shots_used_tmp = []
    total_paulis_evaluated_tmp = []
    circuit_depth_tmp = []
    total_one_qubit_gates_tmp = []
    total_two_qubit_gates_tmp = []

    for i in range(random_runs):

        if i == 0:
            qWF = WaveFunctionCircuit(
            active_space,
            mo_coeff,  # MO coefficients from previous ideal simulator run
            integral_generator,
            QI,  # pass QuantumInterface
            include_active_kappa = True  
            )

            print("#### QI.get_info ####")
            QI.get_info()

            print()
            print("Total one-qubit gates", count_qubit_gates(circuit=qWF.QI.circuit)[0]+active_space[0])
            print("Total two-qubit gates", count_qubit_gates(circuit=qWF.QI.circuit)[1])
            print("Circuit depth", qWF.QI.circuit.depth())
            print("Circuit width (#qubits)", qWF.QI.circuit.width())
            print()

            large_font = {"fontsize": 20,"subfontsize": 8}
            #print(qWF.QI.circuit.decompose().decompose().decompose().draw(output="mpl", fold=50, style=large_font))
            #plt.savefig('non_transpiled_circuit.png')

            # We have adjusted the tolerance and maxiter to looser values because it will be hard to converge shot noise
            qWF.run_wf_optimization_2step("rotosolve", orbital_optimization = True, tol = 1e-3, maxiter = 1)
            #qWF.run_wf_optimization_2step("BFGS", orbital_optimization = True, tol = 1e-3, maxiter = 500)
        
            opt_angles_tmp.append(qWF.thetas)
            opt_c_mo_tmp.append(qWF.c_mo)
            total_shots_used_tmp.append(qWF.QI.total_shots_used)
            total_paulis_evaluated_tmp.append(qWF.QI.total_paulis_evaluated)
            circuit_depth_tmp.append(qWF.QI.circuit.depth())
            total_one_qubit_gates_tmp.append(count_qubit_gates(circuit=qWF.QI.circuit)[0]+active_space[0])
            total_two_qubit_gates_tmp.append(count_qubit_gates(circuit=qWF.QI.circuit)[1])

            print("Total Paulis evaluated",qWF.QI.total_paulis_evaluated)

            if float(qWF.energy_elec + mol.energy_nuc()) < best_energy:
                best_energy = float(qWF.energy_elec + mol.energy_nuc())
                best_run = i
                print("best_energy:",best_energy, "best_run:", best_run, "\n")


        qWF = WaveFunctionCircuit(
        active_space,
        mo_coeff,  # MO coefficients from previous ideal simulator run
        integral_generator,
        QI,  # pass QuantumInterface
        include_active_kappa = True  
        )

        qWF.thetas = (np.random.random(len(qWF.thetas))*2*np.pi).tolist()
        qWF.run_wf_optimization_2step("rotosolve", orbital_optimization = True, tol = 1e-3, maxiter = 200)
        #qWF.run_wf_optimization_2step("BFGS", orbital_optimization = True, tol = 1e-3, maxiter = 500)

        opt_angles_tmp.append(qWF.thetas)
        opt_c_mo_tmp.append(qWF.c_mo)
        total_shots_used_tmp.append(qWF.QI.total_shots_used)
        total_paulis_evaluated_tmp.append(qWF.QI.total_paulis_evaluated)
        circuit_depth_tmp.append( qWF.QI.circuit.depth())
        total_one_qubit_gates_tmp.append(count_qubit_gates(circuit=qWF.QI.circuit)[0]+active_space[0])
        total_two_qubit_gates_tmp.append(count_qubit_gates(circuit=qWF.QI.circuit)[1])

        if float(qWF.energy_elec + mol.energy_nuc()) < best_energy:
            best_energy = float(qWF.energy_elec + mol.energy_nuc())
            best_run = i
            print("best_energy:",best_energy, "best_run:", best_run, "\n")
    

    print("#### QI.get_info ####")
    QI.get_info()




