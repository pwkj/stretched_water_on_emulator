import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns
import json

with open("H2O_4_4_aug-cc-pvdz_tUPS_HFstartorbs_L=2_ideal_tol=1e-3_maxiter=100.json") as f:
    L2_ideal = json.load(f)



