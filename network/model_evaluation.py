import torch
import numpy as np
from network import *

def serial_sample_generator(model, action, *, batch_size, N_samples):
    layers, prior = model['layers'], model['prior']
    x, logq, logp = None, None, None
    for i in range(N_samples):
        batch_i = i % batch_size
        if batch_i == 0:
            # we're out of samples to propose, generate a new batch
            x, logq = apply_flow_to_prior(prior, layers, batch_size=batch_size)
            logp = -action(x)
        yield x[batch_i], logq[batch_i], logp[batch_i]

def make_mcmc_ensemble(model, action, *, batch_size, N_samples):
    history = {
        'x' : [],
        'logq' : [],
        'logp' : [],
        'accepted' : []
    }

    # build Markov chain
    sample_gen = serial_sample_generator(model, action, batch_size=batch_size, N_samples=N_samples)
    for new_x, new_logq, new_logp in sample_gen:
        if len(history['logp']) == 0:
            # always accept first proposal, Markov chain must start somewhere
            accepted = True
        else: 
            # Metropolis acceptance condition
            last_logp = history['logp'][-1]
            last_logq = history['logq'][-1]
            p_accept = torch.exp((new_logp - new_logq) - (last_logp - last_logq))
            p_accept = min(1, p_accept)
            draw = torch.rand(1) # ~ [0,1]
            if draw < p_accept:
                accepted = True
            else:
                accepted = False
                new_x = history['x'][-1]
                new_logp = last_logp
                new_logq = last_logq
        # Update Markov chain
        history['logp'].append(new_logp)
        history['logq'].append(new_logq)
        history['x'].append(new_x)
        history['accepted'].append(accepted)

    return history


def slow_make_mcmc_ensemble(model, action, *, batch_size, N_samples):
    acc = 0
    first = True
    # Initialize
    last_x, last_logq = apply_flow_to_prior(model['prior'], model['layers'], batch_size=1)
    last_logp = -action(last_x)
    new_x, new_logq = apply_flow_to_prior(model['prior'], model['layers'], batch_size=1)
    new_logp = -action(new_x)

    # build Markov chain
    sample_gen = serial_sample_generator(model, action, batch_size=batch_size, N_samples=N_samples)
    for i in range(N_samples):
    # for new_x, new_logq, new_logp in sample_gen:
        if first:
            # always accept first proposal, Markov chain must start somewhere
            last_x, last_logq = apply_flow_to_prior(model['prior'], model['layers'], batch_size=1)
            last_logp = -action(last_x)
            acc += 1
            first = False
        else: 
            new_x, new_logq = apply_flow_to_prior(model['prior'], model['layers'], batch_size=1)
            new_logp = -action(new_x)
            # Metropolis acceptance condition
            p_accept = torch.exp((new_logp - new_logq) - (last_logp - last_logq))
            p_accept = min(1, p_accept)
            draw = torch.rand(1) # ~ [0,1]
            if draw < p_accept:
                acc += 1
            else:
                new_x = last_x
                new_logp = last_logp
                new_logq = last_logq
        # Update Markov chain

    return acc / N_samples
