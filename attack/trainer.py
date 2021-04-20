import torch
import wandb
import numpy as np
import os
from sklearn.model_selection import train_test_split

from attack.unsupervised import generate_chain
from attack.supervised import generate_adv
from thirdparty.pytorch_msssim import msssim
from utils.distributions import gaus_skl


def train(model, dataloader, args):
    # get reference point
    x, y = iter(dataloader).__next__()
    x_trg, x_ref, y_trg, y_ref = train_test_split(x, y, stratify=y, test_size=args.attack.N_ref)

    if args.attack.type == 'supervised':
        _, x_trg, _, y_trg = train_test_split(x_trg, y_trg, stratify=y_trg,
                                              test_size=args.attack.N_trg)
        # train adversarial samples
        train_fn(model, x_ref, args, x_trg)
    else:
        # train adversarial samples
        train_fn(model, x_ref, args)


def train_fn(model, x_ref, args, x_trg=None):
    total_logs = {
        'Av_ref_sim': 0.,
        'Av_ref_rec_sim': 0.,
        'Av_omega': 0.,
        'Similarity_diff': 0.
    }
    if x_trg is not None:
        total_logs['Av_trg_rec_sim'] = 0.

    # save target images
    if x_trg is not None:
        with torch.no_grad():
            trg_recon, _, _, _, _ = model.forward(x_trg)
        trg_recon = trg_recon.reshape(x_trg.shape).detach()
    else:
        trg_recon = None

    for step, xi in enumerate(x_ref):
        xi = xi.unsqueeze(0)
        x_hist = [xi]
        if x_trg is None:
            for _ in range(args.attack.N_chains):
                x_chain = generate_chain(args.attack.N_adv, xi, model, step_size=0.9)
                x_hist.append(torch.cat(x_chain)[1:])
        else:
            x_adv = generate_adv(x_trg, xi, model, args.attack.eps_reg, lr=.1, max_iter=1000)
            x_hist.append(torch.cat(x_adv)[1:])

        x_hist = torch.cat(x_hist)
        torch.save(x_hist, os.path.join(wandb.run.dir, 'x_hist_{}.pth'.format(step)))
        with torch.no_grad():
            x_recon, _, _, _, _ = model.forward(x_hist)
            z, z_logvar = model.q_z(x_hist)
        x_recon = x_recon.reshape(x_hist.shape).detach()
        torch.save(x_recon, os.path.join(wandb.run.dir, 'x_recon_{}.pth'.format(step)))

        logs = save_stats(x_hist.detach(), x_recon.detach(), z, z_logvar, x_trg, trg_recon)

        total_logs['Av_ref_sim'] += logs['Ref similarity']
        total_logs['Av_ref_rec_sim'] += logs['Rec similarity']
        total_logs['Av_omega'] += logs['SKL [q_a | q]']
        if x_trg is not None:
            total_logs['Av_trg_rec_sim'] += logs['Target Rec similarity']
            total_logs['Similarity_diff'] = total_logs['Av_trg_rec_sim'] - total_logs['Av_ref_rec_sim']

        for k in total_logs:
            wandb.run.summary[k] = total_logs[k]/(step+1)


def save_stats(x_hist, x_recon, z, z_logvar, trg=None, trg_recon=None):
    MB = x_hist.shape[0]

    # similarity with x_ref
    ref_sim = [msssim(x_hist[:1], torch.clamp(x_hist[i:i + 1], 0, 1), window_size=6,
                  normalize='relu').data for i in range(1, MB)]
    # similarity with reconstractions
    rec_sim = [msssim(x_recon[:1], x_recon[i:i + 1], window_size=6,
                  normalize='relu').data for i in range(1, MB)]
    # sKL in latent space
    s_kl = gaus_skl(z[:1], z_logvar[:1], z[1:], z_logvar[1:]).mean()
    mus = (z[:1] - z[1:]).pow(2).sum(1).mean()

    logs = {
        'Adversarial Inputs': wandb.Image(batch_min_max_scale(x_hist[1:]), mode='L'),
        'Adversarial Rec': wandb.Image(x_recon[1:], mode='L'),
        'Ref similarity': np.mean(ref_sim),
        'Rec similarity': np.mean(rec_sim),
        'SKL [q_a | q]': s_kl,
        'Mean dist': mus,
    }

    if trg is not None:
        trg_rec_sim = [msssim(trg_recon[i-1:i], x_recon[i:i+1], window_size=6,
                       normalize='relu').data for i in range(1, MB)]
        logs['Target Rec similarity'] = np.mean(trg_rec_sim)
        logs['Target Inputs'] = wandb.Image(trg, mode='L')
        logs['Target Rec'] = wandb.Image(trg_recon, mode='L')

    wandb.log(logs)
    return logs


def batch_min_max_scale(x):
    MB = x.shape[0]
    mi = x.reshape(MB, -1).min(dim=1).values.reshape(MB, 1,1,1)
    ma = x.reshape(MB, -1).max(dim=1).values.reshape(MB, 1,1,1)
    return (x - mi)/(ma - mi)


