import torch
import numpy as np

def magnetization(phi):
    return torch.mean(phi)
