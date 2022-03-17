import numpy as np
from network import *
import torch

# Loss functions
def calc_dkl(logp, logq): # deprecated
    return (logq - logp).mean()  # reverse KL, assuming samples from q

def dkl(logp, logq):
    return (logq - logp).mean()  # reverse KL, assuming samples from q

def dkl_abs(logp, logq):
    return ((logq - logp).abs()).mean()

def dkl_sr(logp, logq):
    return ((logq - logp)**2).mean()

# Training
def train_step(model, action, loss_fn, optimizer, metrics, *, batch_size):
    layers, prior = model['layers'], model['prior']
    optimizer.zero_grad()

    x, logq = apply_flow_to_prior(prior, layers, batch_size=batch_size)
    logp = -action(x)
    loss = loss_fn(logp, logq)
    loss.backward()

    optimizer.step()

    metrics['loss'].append(grab(loss))
    metrics['logp'].append(grab(logp))
    metrics['logq'].append(grab(logq))
    metrics['ess'].append(grab( compute_ess(logp, logq) ))

# Telemetry
def compute_ess(logp, logq):
    logw = logp - logq
    log_ess = 2*torch.logsumexp(logw, dim=0) - torch.logsumexp(2*logw, dim=0)
    ess_per_cfg = torch.exp(log_ess) / len(logw)
    return ess_per_cfg

def print_metrics(history, avg_last_N_epochs, *, era, epoch):
    print(f'== Era {era} | Epoch {epoch} metrics ==')
    for key, val in history.items():
        avgd = np.mean(val[-avg_last_N_epochs:])
        print(f'\t{key} {avgd:g}')
