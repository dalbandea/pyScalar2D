import numpy as np
import torch

def save_model(filename, model, history):
    torch.save(model['layers'].state_dict(), filename+"_model.pth")
    
    loss = [i.item() for i in history['loss']]
    ess  = [i.item() for i in history['ess']]
    logq = [i.mean() for i in history['logq']]
    logp = [i.mean() for i in history['logp']]
    
    hist = np.transpose(np.vstack((loss, ess, logq, logp)))
    np.savetxt(filename+"_history.txt", hist)

def load_model(path, layers):
    layers.load_state_dict(torch.load(path))
