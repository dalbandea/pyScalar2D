import sys
sys.path.append('.')

import numpy as np
import time
import argparse
from network import *
from hmc import *

print(f"TORCH DEVICE: {torch_device}")

parser = argparse.ArgumentParser()
# -n NTRAJ -t TAU -ns NSTEPS
parser.add_argument("-n", "--ntraj", help="Number of trajectories", default=10,
        type=int)
parser.add_argument("-t", "--tau", help="HMC trajectory length", default=1.0,
        type=float)
parser.add_argument("-ns", "--nsteps", help="Number of integration steps",
        default=10, type=int)
args = parser.parse_args()


# Lattice Theory
L = 8
lattice_shape = (L,L)
beta = 0.25
lmbda = 0.5

# NN Model
prior = SimpleNormal(torch.zeros(lattice_shape), torch.ones(lattice_shape))

n_layers = 16
hidden_sizes = [8,8]
kernel_size = 3
layers = make_phi4_affine_layers(lattice_shape=lattice_shape, n_layers=n_layers, 
    hidden_sizes=hidden_sizes, kernel_size=kernel_size)
model = {'layers': layers, 'prior': prior}


# Define lattice action
phi4action = ScalarPhi4ActionBeta(beta=beta, lam=lmbda) # Returns action of a torch array containing configurations. Check network.py

# HMC parameters
tau = args.tau
n_steps = args.nsteps
n_traj = args.ntraj

# Saving file ID
file_ID = "_b"+str(beta)+"_l"+str(lmbda)+"_ns"+str(n_steps)+"_t"+str(tau)+"_mag"

# Load model
load_model("dumps/L8_b0.25_l0.5_E1000_B500_LOSSdkl_model.pth", model['layers'])
model['layers'].eval()

phi = torch.ones((1,L,L))

with open("results/"+file_ID+".txt", "w") as file_object:
    file_object.close()

for i in range(n_traj):
    flow_hmc(phi, model['layers'], phi4action, tau=tau, n_steps=n_steps,
            reversibility=False)
    mag_i = magnetization(phi).item()

    with open("results/"+file_ID+".txt", "a") as file_object:
        file_object.write(str(mag_i)+"\n")
