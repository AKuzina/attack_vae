from ml_collections import ConfigDict


def get_config(type):
    default_config = dict(
        ### EXPERIMENT
        # iter number to evaluate std
        iter=0,
        # input batch size for training and testing
        batch_size=500,
        test_batch_size=500,
        max_epochs=100,
        gpus=1,
        seed=14,
        dataset_name='mnist',
        # If load pretrained model
        resume=False,

        ### OPTIMIZATION
        # learning rate (initial if cheduler is used)
        lr=5e-4,
        lr_factor=0.9,
        lr_patience=10,

        ### NNs
        # 'mlp, conv, (later hierarcy)'
        arc_type='mlp',
        # Hidden dimension of mlp
        h_dim=300,

        ### VAE
        # latent size
        z_dim=40,

        # prior: standard, mog, boost
        prior='standard',
        # type of the loglikelihood for continuous input: logistic, normal, l1
        likelihood='bernoulli',
        beta=1.,
        warmup=0,
        is_k=1000,
    )

    attack_config = dict(
        # one2many for chain and many2one for attack with target
        type='supervised',
        N_ref=10,

        # for chain
        N_chains=0,
        N_adv=0,

        # for attack with target
        N_trg=5,
        eps_reg=0,

    )

    nvae_config = dict(
        chckpt_path='../NVAE/checkpoint/celeba_64.pt',
        dset_path='../NVAE/datasets/celeba64_lmdb',
        connect=1,
        temp=0.8,
        n_samples=5,
        use_perp=True,
        lr=5e-3,
    )

    default_config = {
        'train': ConfigDict(default_config),
        'attack': ConfigDict({'model': default_config,
                             'attack': ConfigDict(attack_config)}),
        'nvae': ConfigDict({'model': ConfigDict(nvae_config),
                 'attack': ConfigDict(attack_config)})

    }[type]

    return default_config