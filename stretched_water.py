import numpy as np
import matplotlib.pyplot as plt

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


data_points = np.arange(-0.1,0.1,0.01)
n_layers = 1
random_runs = 1
shots = 1000
active_space = (4, 4) #(num_elec, num_orbs)

obj_list = []
for delta_r in data_points:
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
	# ideal_simulator = simulator.ideal_simulator(mol, wf = "tUPS", active_space = active_space, n_layers = n_layers, random_runs = random_runs)
	# print(ideal_simulator)

	# with open('H2O_4_4_aug-cc-pvdz_tUPS_L=2_ideal_5runs.txt', 'a') as f:
	# 	f.write(str(delta_r)+' '+ str(ideal_simulator[0])+ '\n')

	## Shotnoise ####
	shot_simulator = simulator.shot_noise(mol, active_space = active_space, 
	shots = shots, n_layers = n_layers, random_runs=random_runs, wf = "tUPS")

	with open('H2O_4_4_aug-cc-pvdz_tUPS_L=1_1000shots_correct_one.txt', 'a') as f:
		f.write(str(delta_r)+' '+ str(shot_simulator)+ '\n')
	



