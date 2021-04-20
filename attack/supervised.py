import torch
import torch.nn as nn
from tqdm import tqdm
from utils.distributions import gaus_skl


def get_opt_perturbation(x_init, x_trg, vae, reg_weight=1., lr=0.1, max_iter=2500):

    with torch.no_grad():
        z_mean, z_logvar = vae.q_z(x_trg)
    eps = nn.Parameter(torch.zeros_like(x_init), requires_grad=True)

    optimizer = torch.optim.Adam([eps], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=False, patience=100, factor=0.5)
    loss_hist = []
    for i in range(max_iter):
        optimizer.zero_grad()
        x = x_init + eps
        q_m, q_logv = vae.q_z(x)
        loss = gaus_skl(q_m, q_logv, z_mean, z_logvar, dim=1) + reg_weight*torch.norm(eps)
        loss.backward()
        optimizer.step()
        loss_hist.append(loss.item())
        scheduler.step(loss)

    return loss_hist, eps, x_init + eps


def generate_adv(all_trg, x_init, vae, reg_weight, lr=.1, max_iter=300):
    x_inits = [x_init]
    for x_trg in tqdm(all_trg):
        x_trg = x_trg.unsqueeze(0)
        _, eps, x_opt = get_opt_perturbation(x_init, x_trg, vae, reg_weight=reg_weight,
                                             lr=lr, max_iter=max_iter)
        # eps = eps.cpu() / torch.norm(eps.cpu())
        # x_opt = min_max_scale(x_inits[-1] + eps)
        x_inits.append(x_opt.detach().cpu())
    return x_inits

def min_max_scale(x):
    return (x - x.min())/(x.max() - x.min())