print("###################")
print("#### flow HMC #####")
print("###################")

import sys
sys.path.append('.')

import numpy as np
import time
import argparse
from network import *
from hmc import *

print(f"TORCH DEVICE: {torch_device}")


parser = argparse.ArgumentParser()
# python3 test/flow_hmc-test.py -n NTRAJ -t TAU -ns NSTEPS
parser.add_argument("-n", "--ntraj", help="Number of trajectories", type=int, required=True)
parser.add_argument("-t", "--tau", help="HMC trajectory length", default=1.0, type=float)
parser.add_argument("-ns", "--nsteps", help="Number of integration steps", default=10, type=int)
parser.add_argument("-m", "--model", help="Path to pytorch model", type=str, required=True)
args = parser.parse_args()


def parse_info(info, c):
    return info.split(c)[1].split("_")[0]

model_name = args.model.split("/")[1]

# Lattice Theory
L = int(parse_info(model_name, "L"))
lattice_shape = (L,L)
beta = float(parse_info(model_name, "b"))
lmbda = float(parse_info(model_name, "l"))

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
file_ID = "L"+str(L)+"_b"+str(beta)+"_l"+str(lmbda)+"_ns"+str(n_steps)+"_t"+str(tau)+"_mag"

# Load model
load_model(args.model, model['layers'])
model['layers'].eval()

print("###### INFO #####")
print("L    =   ", lattice_shape)
print("beta =   ", phi4action.beta)
print("lam  =   ", phi4action.lam)
print("model=   ", args.model)
print("save =   ", file_ID)
print("tau  =   ", tau)
print("nstep=   ", n_steps)
print("ntraj=   ", n_traj)
print("###### INFO #####")

phi = torch.ones((1,L,L))

with open("results/"+file_ID+".txt", "w") as file_object:
    file_object.close()

for i in range(n_traj):
    flow_hmc(phi, model['layers'], phi4action, tau=tau, n_steps=n_steps,
            reversibility=False)
    mag_i = magnetization(phi).item()

    with open("results/"+file_ID+".txt", "a") as file_object:
        file_object.write(str(mag_i)+"\n")
