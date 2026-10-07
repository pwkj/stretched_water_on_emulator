import numpy as np
import matplotlib.pyplot as plt
import json

import pyscf
from pyscf import scf, mcscf

import slowquant.SlowQuant as sq

import simulator


def stretch_water(delta_r):
	"""
	Stretch the O-H bonds while preserving the H-O-H angle.

	delta_r = 0 -> equilibrium geometry.
	"""

	#equilibrium geometry.
	O  = [0.0000, 0.0000, 0.0000]
	H1 = [0.957848, 0.0000, 0.0000]
	H2 = [-0.2399872, 0.927297, 0.0000]

	# Stretch geometry.
	H1_new = [0.957848 + delta_r, 0.0000, 0.0000]

	# 2q^2 + q*(2x_eq + 2y_eq) + x_eq^2 + y_eq^2 - (x_eq + delta_r)^2 = 0
	a = 2
	b = -2*0.2399872 + 2*0.927297
	c = 0.2399872**2 + 0.927297**2 - 1.0*(0.957848 + delta_r)**2
	q_plus = (-1.0*b +  np.sqrt(b**2 - 4*a*c))/(2*a)

	H2_new = [-0.2399872+q_plus, 0.927297+q_plus, 0.0000]

	assert np.abs(np.linalg.norm(H1_new) - np.linalg.norm(H2_new)) < 1e-6, "np.abs(np.linalg.norm(H1_new) - np.linalg.norm(H2_new))"

	return H1_new, H2_new


#PES_datapoints = np.arange(-0.1,0.1,0.01)
PES_datapoints = [-0.1]
n_layers = 1
runs = 1
shots = 50000
active_space = (4, 4) #(num_elec, num_orbs)

for delta_r in PES_datapoints:
	H1_new, H2_new = stretch_water(delta_r)

	mol = pyscf.gto.M()
	mol.atom = [
	    ['O', [0., 0., 0.]],
	    ['H', H1_new],
	    ['H', H2_new],
	    ]

	mol.basis = "aug-cc-pvdz"
	mol.unit="angstrom"
	mol.build()

	# #### Ideal ####
	ideal_simulator = simulator.ideal_simulator(mol, wf = "tUPS", active_space = active_space, n_layers = n_layers, runs = runs)
	results = {"energies": ideal_simulator}

	with open("H2O_4_4_aug-cc-pvdz_tUPS_L=1_ideal_tol_1e-3_maxiter_200.json", "w") as f:
		json.dump(results, f)


	# with open('H2O_4_4_aug-cc-pvdz_tUPS_L=2_ideal_5runs.txt', 'a') as f:
	# 	f.write(str(delta_r)+' '+ str(ideal_simulator[0])+ '\n')

	## Shotnoise ####
	# shot_noise = simulator.shot_noise_simulator(mol,  wf = "tUPS", n_layers = n_layers,
	# active_space = active_space, shots = shots, runs=runs)

	# results = {
    # "energies": shot_noise[0],
    # "total_shots_used": shot_noise[1],
    # "total_paulis_evaluated": shot_noise[2]}

	# with open("H2O_4_4_aug-cc-pvdz_tUPS_L=1_50000shots_tol_1e-5_maxiter_200.json", "w") as f:
	# 	json.dump(results, f)



