import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

im_criterion = nn.BCEWithLogitsLoss(reduction='sum')
transition_criterion = nn.MSELoss(reduction='sum')

causal_criterion = nn.MSELoss(reduction='none')

def energy(state, action, next_state, pred_next_state=None, no_trans=False,
            sigma=0.5):
    norm = 0.5 / (sigma ** 2)

    if no_trans:
        diff = state - next_state
    else:
        diff = pred_next_state - next_state

    return norm * diff.pow(2).sum(2).mean(1)

def contrastive_loss(state, action, next_state, pred_next_state, 
                        hinge=1., sigma=0.5):
    batch_size = state.size(0)
    if len(state.shape) == 2:
        state = state.view(batch_size, 1, -1)
        next_state = next_state.view(batch_size, 1, -1)
        pred_next_state = pred_next_state.view(batch_size, 1, -1)

    perm = np.random.permutation(batch_size)

    neg_state = state[perm]

    pos_loss = energy(state, action, next_state, pred_next_state, sigma=sigma)
    zeros = torch.zeros_like(pos_loss)
    pos_loss = pos_loss.mean()

    neg_loss = torch.max(zeros, hinge - 
        energy(state, action, neg_state, no_trans=True, sigma=sigma)).mean()

    loss = pos_loss + neg_loss

    return loss

def image_loss(recon, target):
    return im_criterion(recon, target)

def kl_loss(mu, logvar):
    batch_size = mu.shape[0]
    mu = mu.view(batch_size, -1)
    logvar = logvar.view(batch_size, -1)
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def transition_loss(pred_state, state):
    return transition_criterion(pred_state, state)

def causal_loss(pred_states, state, gammaexp, gamma):
    loss = []
    gammagrads = []
    for i, p in enumerate(pred_states):
        mse_loss = causal_criterion(p, state)
        mse_loss = torch.stack(tuple(mse_loss.mean(-1).mean(0)))
        loss.append(mse_loss)
        gammagrads.append(gamma.sigmoid() - gammaexp[i])
    loss = torch.stack(loss)
    dRdgamma = torch.zeros(gamma.size()).to(gamma.device)
    gammagrads = torch.stack(gammagrads)
    norm_loss = loss.softmax(0)
    dRdgamma = torch.einsum("kij,ki->ij", gammagrads, norm_loss)
    return loss.mean(), dRdgamma

