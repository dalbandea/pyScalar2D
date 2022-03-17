import numpy as np
import torch
import argparse
import time
import gc

import sys
sys.path.append('.')
from network import *

print(f"TORCH DEVICE: {torch_device}")

parser = argparse.ArgumentParser()
# -L 8 -l 0.5 -b 0.25 -N 1280000 -B 64 -E 400 -LO "dkl_abs" -w "dumps/"
parser.add_argument("-L", "--lsize", help="Lattice size", default=8, type=int, required=True)
parser.add_argument("-l", "--lam", help="Lambda value", default=8, type=float, required=True)
parser.add_argument("-b", "--beta", help="Beta", type=float)
parser.add_argument("-m2", "--mass2", help="Mass squared", type=float)

parser.add_argument("-B", "--batch", help="Batch size", default=64, type=int)
parser.add_argument("-N", "--nconfigs", help="Number of configurations used for training", required=True, type=int)
parser.add_argument("-LO", "--loss", help="Loss function", choices=['dkl', 'dkl_abs', 'dkl_sr'], default='dkl', type=str)

parser.add_argument("-w", "--writedir", help="Writing directory for saving data", default='dkl', type=str, required=True)
args = parser.parse_args()

writedir = args.writedir
# Lattice Theory
L = args.lsize
lattice_shape = (L,L)
if args.mass2 is not None:
    phi4_action = ScalarPhi4Action(M2=args.mass2, lam=args.lam)
else:
    phi4_action = ScalarPhi4ActionBeta(beta=args.beta, lam=args.lam)


# Model
prior = SimpleNormal(torch.zeros(lattice_shape), torch.ones(lattice_shape))
n_layers = 16
hidden_sizes = [8,8]
kernel_size = 3
layers = make_phi4_affine_layers(lattice_shape=lattice_shape,
        n_layers=n_layers, hidden_sizes=hidden_sizes, kernel_size=kernel_size)
model = {'layers': layers, 'prior': prior}

# Optimizer
base_lr = .01
optimizer = torch.optim.Adam(model['layers'].parameters(), lr=base_lr)
loss_fn = globals()[args.loss]

# Main training setup and loop
batch_size = args.batch
N_epoch = int(args.nconfigs/batch_size)
print_freq = N_epoch
MCMC_eval_freq = 100
MCMC_ensemble_size = 100

history = {
    'loss' : [],
    'logp' : [],
    'logq' : [],
    'ess' : []
}

try:
    save_filename = "L"+str(L)+"_b"+str(phi4_action.beta)+"_l"+str(phi4_action.lam)+"_E"+str(N_epoch)+"_B"+str(batch_size)+"_LOSS"+loss_fn.__name__.replace("_","")
except:
    save_filename = "L"+str(L)+"_b"+str(phi4_action.M2)+"_l"+str(phi4_action.lam)+"_E"+str(N_epoch)+"_B"+str(batch_size)+"_LOSS"+loss_fn.__name__.replace("_","")


print("========= INFO =========")
print("Lattice size:    ", lattice_shape)
if args.mass2 is not None:
    print("Mass^2       =   ", phi4_action.M2)
else:
    print("Beta         =   ", phi4_action.beta)
print("Lambda       =   ", phi4_action.lam)
print("Batch size   =   ", batch_size)
print("Epochs       =   ", N_epoch)
print("Loss         =   ", loss_fn.__name__)
print("Writedir     =   ", writedir)
print("File prefix  =   ", save_filename)
print("========= INFO =========")

# Main loop for training
for epoch in range(N_epoch+1):
    train_step(model, phi4_action, loss_fn, optimizer, history, batch_size=batch_size)

    if epoch % 100 == 0:
        print("Epoch ", epoch, " of ", N_epoch)
        # Save data
        save_model(writedir+save_filename, model, history)
