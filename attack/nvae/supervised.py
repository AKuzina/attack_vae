import torch
import torch.nn as nn
from tqdm import tqdm
from IPython.display import clear_output
import matplotlib.pylab as plt

from attack.nvae.utils import VGGPerceptualLoss
from utils.distributions import gaus_skl


def get_opt_perturbation(x_init, x_trg, model, connect=0, reg_weight=1., lr=0.1, max_iter=2500, use_perp=True):
    eps = nn.Parameter(torch.zeros_like(x_init), requires_grad=True)

    with torch.no_grad():
        _, q_dist_trg, _, _, _, _ = model(x_trg, connect=connect)

    optimizer = torch.optim.SGD([eps], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=10, factor=0.5)
    perp_loss = VGGPerceptualLoss(x_init)
    perp_loss.to('cuda')
    loss_hist = []
    reg_hist = []
    # clear_output(wait=True);
    for i in range(max_iter):
        optimizer.zero_grad()
        x = torch.clamp(x_init + eps, 0., 1.)
        _, q_dist_curr, _, _, _, _ = model(x, connect=connect)
        # compute loss
        if use_perp:
            reg = perp_loss(x) + torch.norm(eps)
        else:
            reg = torch.norm(eps)
        loss = sup_loss(q_dist_trg, q_dist_curr) + reg_weight*reg

        loss.backward()
        optimizer.step()
        loss_hist.append(loss.item())
        reg_hist.append(reg.item())
        scheduler.step(loss)
        if optimizer.param_groups[0]["lr"] < 1e-6:
            break
    # fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
    # ax[0].plot(loss_hist)
    # ax[0].set_yscale('log')
    # ax[0].set_title('Loss')
    # ax[1].plot(reg_hist)
    # ax[1].set_title('Regularization')

    return loss_hist, eps, x_init + eps


def sup_loss(q_dist_trg, q_dist_curr):
    loss = 0.
    for q_1, q_2 in zip(q_dist_trg, q_dist_curr):
        sym_kl = gaus_skl(q_1.mu, 2*torch.log(q_1.sigma), q_2.mu, 2*torch.log(q_2.sigma))
        loss += sym_kl.sum()
    return loss


def generate_adv(all_trg, x_init, model, connect, reg_weight, lr=.1, max_iter=300, use_perp=True):
    x_inits = [x_init.cpu()]
    for x_trg in tqdm(all_trg):
        x_trg = x_trg.unsqueeze(0)
        _, eps, x_opt = get_opt_perturbation(x_init.cuda(), x_trg.cuda(), model, connect=connect,
                                             reg_weight=reg_weight, lr=lr, max_iter=max_iter,
                                             use_perp=use_perp)
        x_inits.append(x_opt.detach().cpu())
    return x_inits


def min_max_scale(x):
    return (x - x.min())/(x.max() - x.min())