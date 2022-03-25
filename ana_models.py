import sys
sys.path.append(".")

import os
import numpy as np
import matplotlib.pyplot as plt
from network import *


def ana_acc(lambdas, batches, wdir, phi4_action, lattice_shape):
    dir_lst = list(os.walk(wdir))[0][2]
    for lam in lambdas:
        print("######### lam = ", lam, "#########")
        for batch in batches:
            datafiles = [k for k in dir_lst if ("model" in k and lam in k and batch in k)]
            print("· Batch = ", batch)
            for datafile in datafiles:
                acc = get_acceptance(wdir+datafile, phi4_action, lattice_shape)
                print("     Acceptance = ", acc)

def get_acceptance(datafile, phi4_action, lattice_shape):
    # Load model
    prior = SimpleNormal(torch.zeros(lattice_shape), torch.ones(lattice_shape))
    n_layers = 16
    hidden_sizes = [8,8]
    kernel_size = 3
    layers = make_phi4_affine_layers(lattice_shape=lattice_shape,
            n_layers=n_layers, hidden_sizes=hidden_sizes, kernel_size=kernel_size)
    model = {'layers': layers, 'prior': prior}

    load_model(datafile, model['layers'])
    model['layers'].eval()

    ensemble_size = 1000
    phi4_ens = make_mcmc_ensemble(model, phi4_action, batch_size=64, N_samples=ensemble_size)
    meanacc = np.mean(phi4_ens["accepted"])
    return meanacc, np.std(phi4_ens["accepted"])/np.sqrt(len(phi4_ens["accepted"]))

def read_file(filepath):
    data = np.loadtxt(filepath)
    loss = list(data[:,0])
    ess = list(data[:,1])
    return loss, ess

def ana_files(*args, wdir):
    dir_lst = list(os.walk(wdir))[0][2]
    losses  = []
    esss    = []
    filepaths = []
    for arg in itertools.product(*args):
        print(str(arg))
        for filepath in filterdir(arg, wdir):
            print(filepath)
            loss, ess = read_file(filepath)
            losses.append(loss)
            esss.append(ess)
            filepaths.append(filepath)
    plot_batch(losses, filepaths, "loss")
    plot_batch(esss, filepaths, "ess")

def plot_batch(data, labels, title):
    for i in range(len(data)):
        plt.plot(range(len(data[i])), data[i], label=labels[i])
    plt.title(title)
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

def print_matching_files(*args, wdir):
    dir_lst = list(os.walk(wdir))[0][2]
    filepaths = []
    for arg in itertools.product(*args):
        for filepath in filterdir(arg, wdir):
            filepaths.append(filepath)
    return filepaths

def filterdir(texts, wdir):
    dir_lst = list(os.walk(wdir))[0][2]
    filtered_lst = dir_lst.copy()
    for text in texts:
        filtered_lst = [k for k in filtered_lst if text in k]
    return filtered_lst

working_dir = "dumps/"
lambdas = ['l'+str(i) for i in [0.5]]
betas   = ['b'+str(i)+'_' for i in [0.586, 0.6, 0.65, 0.7]]
Ls      = ['b'+str(i) for i in [8]]

# Analyze loss and ess
## Beta parametrization

test = ['L8_b0.586_l0.5_E5000_B250_LOSSdkl_history.txt',
 'L8_b0.6_l0.5_E5000_B250_LOSSdkl_history.txt',
 'L8_b0.65_l0.5_E5000_B250_LOSSdkl_history.txt',
 'L8_b0.7_l0.5_E5000_B250_LOSSdkl_history.txt']


ana_files(test, wdir=".")

## Mass parametrization
ana_data(['l8.0'], ['B250', 'B500', 'B1000', 'B2000'], working_dir)


# Analyze acceptance
## Mass parametrization
L = 8
lattice_shape = (L,L)
M2 = -4.0
lam = 8.0
phi4_action = ScalarPhi4Action(M2=M2, lam=lam)
ana_acc(['l8.0'], batches, working_dir, phi4_action, lattice_shape)

## Beta parametrization
L = 8
lattice_shape = (L,L)
beta = 0.7
lam = 0.5
phi4_action = ScalarPhi4ActionBeta(beta=beta, lam=lam)
# ana_acc(['l0.5'], batches, working_dir, phi4_action, lattice_shape)

accs = []
for i in tqdm(range(500)):
    acc, accstd = get_acceptance('L8_b0.7_l0.5_E5000_B250_LOSSdkl_model.pth', phi4_action, lattice_shape)
    accs.append(acc)
