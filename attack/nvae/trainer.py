import torch
import wandb
import numpy as np
import os
from sklearn.model_selection import train_test_split

import NVAE.utils
from attack.nvae.supervised import generate_adv, sup_loss
from thirdparty.pytorch_msssim import msssim
from utils.distributions import gaus_skl


def train(model, dataloader, args):
    # get reference point
    x, _ = iter(dataloader).__next__()
    x_trg, x_ref = train_test_split(x, test_size=args.attack.N_ref, random_state=1234)

    if args.attack.type == 'supervised':
        _, x_trg = train_test_split(x_trg, test_size=args.attack.N_trg, random_state=1234)
        # train adversarial samples
        train_fn(model, x_ref, args, x_trg)
    # else:
        # train adversarial samples
        # train_fn(model, x_ref, args)


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
            logits, _, _, _, _, _ = model(x_trg.cuda(), connect=args.model.connect, t=args.model.temp)
            trg_recon = model.decoder_output(logits).sample()
    else:
        trg_recon = None
    for step, xi in enumerate(x_ref):
        xi = xi.unsqueeze(0)
        x_hist = [xi]
        if x_trg is not None:
            x_adv = generate_adv(x_trg, xi, model, connect=args.model.connect,
                                 reg_weight=args.attack.eps_reg, lr=args.model.lr,
                                 max_iter=800,
                                 use_perp=args.model.use_perp)
            x_hist.append(torch.cat(x_adv)[1:])

        x_hist = torch.cat(x_hist)
        torch.save(x_hist, os.path.join(wandb.run.dir, 'x_hist_{}.pth'.format(step)))

        with torch.no_grad():
            logits, q_dist_hist, _, _, _, _ = model(x_hist[1:].cuda(), connect=args.model.connect,
                                                    t=args.model.temp)
            x_a_recon = model.decoder_output(logits).sample()
            logits, q_dist_ref, _, _, _, _ = model(x_hist[:1].cuda(), connect=args.model.connect,
                                                    t=args.model.temp)
            x_r_recon = model.decoder_output(logits).sample()


        torch.save(x_a_recon.cpu(), os.path.join(wandb.run.dir, 'x_recon_{}.pth'.format(step)))

        logs = save_stats(x_hist.detach(), x_a_recon.detach(), x_r_recon.detach(), q_dist_hist, q_dist_ref, model, x_trg, trg_recon)

        total_logs['Av_ref_sim'] += logs['Ref similarity']
        total_logs['Av_ref_rec_sim'] += logs['Rec similarity']
        total_logs['Av_omega'] += logs['SKL [q_a | q]']
        if x_trg is not None:
            total_logs['Av_trg_rec_sim'] += logs['Target Rec similarity']
            total_logs['Similarity_diff'] = total_logs['Av_trg_rec_sim'] - total_logs['Av_ref_rec_sim']

        for k in total_logs:
            wandb.run.summary[k] = total_logs[k]/(step+1)


def save_stats(x_hist, x_a_recon, x_r_recon, q_dist_hist, q_dist_ref, model, trg=None, trg_recon=None):
    MB = x_hist.shape[0]

    # similarity with x_ref
    ref_sim = [msssim(x_hist[:1], torch.clamp(x_hist[i:i + 1], 0, 1), normalize='relu').data for i in range(1, MB)]
    # sKL in latent space

    s_kl = skl(q_dist_ref, q_dist_hist).cpu().data
    # print(s_kl)
    # similarity with reference (rec)
    rec_sim = [msssim(x_hist[:1].cpu(), x_a_recon[i-1:i].cpu(), normalize='relu').data for i in range(1, MB)]

    # list of ELBO_k
    # elbos = elbo_k(x_hist, model)

    logs = {
        'Adversarial Inputs': wandb.Image(x_hist[1:]),
        'Adversarial Rec': wandb.Image(torch.clamp(x_a_recon, 0, 1)),
        'Ref similarity': np.mean(ref_sim),
        'Rec similarity': np.mean(rec_sim),
        'SKL [q_a | q]': s_kl, #np.mean(s_kl),
    }
    # for i in range(elbos.shape[1]):
    #     print(elbos[:, i])
    #     logs['ELBO_{}'.format(i)] = torch.mean(elbos[:, i]).data.cpu()

    if trg is not None:
        trg_rec_sim = [msssim(trg[i-1:i].cpu(), x_a_recon[i-1:i].cpu(),
                       normalize='relu').data for i in range(1, MB)]
        logs['Target Rec similarity'] = np.mean(trg_rec_sim)
        logs['Target Inputs'] = wandb.Image(trg)
        logs['Target Rec'] = wandb.Image(trg_recon)

    wandb.log(logs)
    return logs


def skl(q_ref, q_adv):
    loss = 0.
    for q_1, q_2 in zip(q_ref, q_adv):
        # print(q_1.mu.shape, q_2.mu.shape)
        sym_kl = gaus_skl(q_1.mu, 2*torch.log(q_1.sigma), q_2.mu, 2*torch.log(q_2.sigma), (1,2,3))
        loss += sym_kl.mean()
    return loss


def batch_min_max_scale(x):
    MB = x.shape[0]
    mi = x.reshape(MB, -1).min(dim=1).values.reshape(MB, 1,1,1)
    ma = x.reshape(MB, -1).max(dim=1).values.reshape(MB, 1,1,1)
    return (x - mi)/(ma - mi)


def elbo_k(x, model):
    x = torch.clamp(x, 0, 1).cuda()
    N_max = 35
    res = []
    for i in range(N_max):
        with torch.no_grad():
            logits, _, all_q, all_p, kl_all, _ = model(x, connect=i)
        output = model.decoder_output(logits)
        recon_loss = NVAE.utils.reconstruction_loss(output, x, crop=model.crop_output)
        balanced_kl, _, _ = NVAE.utils.kl_balancer(kl_all, kl_balance=False)
        nelbo = recon_loss + balanced_kl
        # bits per dim
        bpd_coeff = 1. / np.log(2.) / (3 * 64 * 64)
        nelbo = nelbo*bpd_coeff
        res.append(nelbo.cpu())
    return torch.stack(res, 1)