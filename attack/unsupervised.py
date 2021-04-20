import torch
import torch.nn as nn
from tqdm import tqdm


def get_jac_enc(x, vae):
    x.requires_grad = True
    z_mean, z_logvar = vae.q_z(x)
    z_mean = z_mean.squeeze(0)
    jac = [torch.autograd.grad(z, x, retain_graph=True)[0]   for z in z_mean]
    return torch.stack(jac)


def get_best_direction(x_init, vae, lr=1e-3, max_iter=1000):
    eps = torch.randn_like(x_init)*0.2
    eps.requires_grad = True
    x_dim = x_init.shape[-1]*x_init.shape[-2]*x_init.shape[-3]
    loss_hist = []
    J = get_jac_enc(x_init, vae)
    J = J.reshape(-1, x_dim)
    for i in range(max_iter):
        J_eps = J @ eps.reshape(x_dim)
        loss = -torch.norm(J_eps) # + torch.norm(eps)
        grad = torch.autograd.grad(loss, eps, create_graph=True)[0]
        eps = eps - lr/(i//2+1)*grad
        loss_hist.append(loss.item())
        if torch.norm(grad) < 1e-3:
            print('break after {} iterations'.format(len(loss_hist)))
            break

    return loss_hist, eps, eps+x_init


def generate_chain(N_steps, x_init, vae, step_size=1., lr=10., max_iter=300):
    x_inits = [x_init]

    for i in tqdm(range(N_steps)):
        loss_hist, eps, x_opt = get_best_direction(x_inits[i], vae, lr=lr, max_iter=max_iter)
        eps = eps.cpu() / torch.norm(eps.cpu())
        x_opt = min_max_scale(x_inits[i] + step_size * eps)
        x_inits.append(x_opt.detach().cpu())
        # TODO acceptance test

    return x_inits


def min_max_scale(x):
    return (x - x.min())/(x.max() - x.min())