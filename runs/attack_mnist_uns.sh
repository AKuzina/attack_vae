#!/usr/bin/env bash


for beta in 0.5  1 2 4 10
do
python run_attack.py \
            --config.model.dataset_name='fashion_mnist'\
            --config.model.batch_size=256\
            --config.model.test_batch_size=1024\
            --config.model.z_dim=128\
            --config.model.beta=$beta\
            --config.model.arc_type='conv'\
            --config.model.h_dim=500 \
            --config.model.max_epochs=500 \
            --config.model.warmup=0 \
            --config.model.lr=5e-4\
            --config.model.prior="standard"\
            --config.attack.type='unsupervised'\
            --config.attack.N_ref=50\
            --config.attack.N_adv=2\
           --config.attack.N_chains=3
done