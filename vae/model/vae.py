import math
import numpy as np
import torch
import torch.nn as nn
import wandb
import pytorch_lightning as pl

from utils.distributions import log_Bernoulli, log_Gaus_diag
from vae.model.priors import StandardNormal, RealNPV
from vae.utils.architecture import get_architecture


class VAE(nn.Module):
    def __init__(self, architecture, likelihood, prior):
        super(VAE, self).__init__()
        assert likelihood in ['bernoulli', 'gaussian'], \
            'unknown likelihood type {}'.format(likelihood)

        self.likelihood = likelihood
        self.prior = prior

        self.encoder, self.decoder = architecture()
        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or \
                    isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)

    def q_z(self, x):
        """
        Encoder
        :param x: input image
        :return: parameters of q(z|x), (MB, z_dim)
        """
        z_q_mean, z_q_logvar = self.encoder(x)
        return z_q_mean, z_q_logvar

    def p_x(self, z):
        """
        Decoder
        :param z: latent vector          (MB, z_dim)
        :return: parameters of p(x|z)    (MB, inp_dim)
        """
        x_mean, x_logvar = self.decoder(z)
        return x_mean, x_logvar

    def forward(self, x):
        # z ~ q(z | x)
        z_q_mean, z_q_logvar = self.q_z(x)
        z_q = self.reparametrize(z_q_mean, z_q_logvar)
        x_mean, x_logvar = self.p_x(z_q)

        # reshape for convolutional architectures
        x_mean = x_mean.reshape(x_mean.shape[0], -1)
        x_logvar = x_logvar.reshape(x_mean.shape[0], -1)
        # print(z_q_mean.min(),z_q_mean.max())
        # print(z_q_logvar.min(), z_q_logvar.max())
        return x_mean, x_logvar, z_q, z_q_mean, z_q_logvar

    def reconstruct_x(self, x):
        x_mean, _, _, _, _ = self.forward(x)
        return x_mean

    def reconstruction_error(self, x, x_mean, x_logvar):
        if self.likelihood == 'bernoulli':
            re = -log_Bernoulli(x, x_mean, dim=1)
        elif self.likelihood == 'gaussian':
            re = -log_Gaus_diag(x, x_mean, x_logvar, dim=1)
        return re

    def kl(self, z, z_mean, z_logvar):
        """
        KL-divergence between p(z) and q(z|x)
        :param z:           (MB, z_dim)
        :param z_mean:      (MB, z_dim)
        :param z_logvar:    (MB, z_dim)
        :return: KL         (MB, )
        """
        log_p_z = self.prior.log_prob(z)
        log_q_z = log_Gaus_diag(z, z_mean, z_logvar, dim=1)
        # print('prior', log_p_z.sum().item())
        # print('q', log_q_z.sum().item())
        kl_value = log_q_z - log_p_z
        return kl_value

    def ELBO(self, x, beta=1., average=False):
        """
        :param x:   (MB, inp_dim)
        :param beta: Float
        :param average: Compute average over mini batch or not, bool
        :return: Re + beta * KL
        """
        x_mean, x_logvar, z_q, z_q_mean, z_q_logvar = self.forward(x)
        MB = x.shape[0]

        # data term
        re = self.reconstruction_error(x.view(MB, -1), x_mean, x_logvar)
        # KL-divergence
        kl = self.kl(z_q, z_q_mean, z_q_logvar)

        loss = re + beta * kl

        if average:
            loss = torch.mean(loss, 0)
            re = torch.mean(re, 0)
            kl = torch.mean(kl, 0)

        return loss, re, kl

    def estimate_nll(self, X, K=1000):
        """
        Estimate NLL by importance sampling
        :param X: mini-batch, (N, x_dim(s))
        :param samples: Samples per observation
        :return: IS estimate
        """
        N = X.shape[0]
        mu_z, logvar_z = self.q_z(X)  # -> (N, z_dim)
        total_nll = []

        for j in range(N):
            mu_z_curr, logvar_z_curr = mu_z[j:j+1].repeat(K, 1), logvar_z[j:j+1].repeat(K, 1)

            # for r in range(0, rep):
            z_q = self.reparametrize(mu_z_curr, logvar_z_curr)
            x_mean, x_logvar = self.p_x(z_q)
            log_p_x = -self.reconstruction_error(X[j:j+1].view(1, -1),
                                                 x_mean.reshape(K, -1),
                                                 x_logvar.reshape(K, -1))
            log_p_z = self.prior.log_prob(z_q)
            log_q_z = log_Gaus_diag(z_q, mu_z_curr, logvar_z_curr, dim=-1)
            ll_curr = log_p_x + log_p_z - log_q_z
            ll_obs = torch.logsumexp(ll_curr, 0) - np.log(K)
            total_nll.append(-ll_obs)
        return torch.tensor(total_nll)

    def generate_x(self, N=25):
        z_sample =self.prior.sample_n(N)
        x_mean, _ = self.p_x(z_sample)
        return x_mean

    @staticmethod
    def reparametrize(mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(mu.device)
        return eps.mul(std).add_(mu)


class StandardVAE(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        arc, hparams = get_architecture(hparams)
        if hparams.prior == 'standard':
            prior = StandardNormal(hparams.z_dim)
        elif hparams.prior == 'realnvp':
            prior = RealNPV(hparams.z_dim)

        self.vae = VAE(arc, hparams.likelihood, prior)
        self.params = hparams
        self.save_hyperparameters()
        self.x_rec = None

    def forward(self, x):
        return self.vae(x)

    def training_step(self, batch, batch_idx):
        beta = self.params.beta
        if self.params.warmup > 0:
            beta *= min(1, self.current_epoch/self.params.warmup)

        loss, re, kl = self.vae.ELBO(batch[0], beta=beta, average=True)
        # logging
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True,
                 logger=True)
        self.log('train_kl', kl.detach(), on_step=True, on_epoch=True, prog_bar=False,
                 logger=True)
        self.log('train_re', re.detach(), on_step=True, on_epoch=True, prog_bar=False,
                 logger=True)
        self.log('beta', beta, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, re, kl = self.vae.ELBO(batch[0], beta=self.params.beta, average=True)
        # logging
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True,
                 logger=True)
        self.log('val_kl', kl.detach(), on_step=False, on_epoch=True, prog_bar=False,
                 logger=True)
        self.log('val_re', re.detach(), on_step=False, on_epoch=True, prog_bar=False,
                 logger=True)
        # reconstructions
        if self.x_rec is None:
            self.x_rec = batch[0]

    def validation_epoch_end(self, outputs):
        # samples
        sample = self.vae.generate_x(9)
        sample = sample.reshape(9, self.params.image_size[0], self.params.image_size[1], -1).detach()
        self.log('Prior_sample', wandb.Image(sample))

        # reconstructions
        plot_rec = self.vae.reconstruct_x(self.x_rec[:9])
        plot_rec = plot_rec.reshape(9, self.params.image_size[0],
                                  self.params.image_size[1], -1).detach()
        self.log('Reconstructions', wandb.Image(plot_rec))

        # latent space
        z_mean, _ = self.vae.q_z(self.x_rec)
        dta = [[x[0], x[1]] for x in z_mean]
        table = wandb.Table(data=dta, columns=["x", "y"])
        logger = self.logger.experiment
        logger.log({'Latent_space':  wandb.plot.scatter(table, "x", "y")})

    def test_step(self, batch, batch_idx):
        # elbo
        loss, re, kl = self.vae.ELBO(batch[0], beta=self.params.beta, average=True)
        # IWAE
        nll = self.vae.estimate_nll(batch[0], self.params.is_k)
        # logging
        self.log('test_elbo', -loss.detach(), on_step=True, on_epoch=False, prog_bar=False,
                 logger=True)
        return {'nll':nll, 'labels':batch[1]}

    def test_epoch_end(self, outputs):
        nll = torch.cat([x['nll'] for x in outputs]).data.cpu()
        labels = torch.cat([x['labels'] for x in outputs]).data.cpu()

        # NLL on the whole test set
        self.log('test_nll', nll.mean())
        logger = self.logger.experiment
        # Per task eval
        for l in np.unique(labels):
            idx = np.where(labels == l)[0]
            data = nll[idx]
            for i in range(len(data)):
                logger.log({'test_nll_task{}'.format(l): data[i]})

        # samples
        N = 100
        sample = self.vae.generate_x(N)
        sample = sample.reshape(N, self.params.image_size[0], self.params.image_size[1], -1).detach()
        self.log('Samples', wandb.Image(sample))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.params.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               factor=self.params.lr_factor,
                                                               patience=self.params.lr_patience)
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
            'monitor': 'val_loss'
        }