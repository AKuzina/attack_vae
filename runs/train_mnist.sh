#!/usr/bin/env bash


for beta in 0.5 1 2 4 10
do
python run_experiment.py \
            --config.dataset_name='fashion_mnist'\
            --config.batch_size=256\
            --config.test_batch_size=1024\
            --config.z_dim=128\
            --config.beta=$beta\
            --config.arc_type='conv'\
            --config.h_dim=500 \
            --config.max_epochs=500 \
            --config.warmup=0 \
            --config.lr=5e-4\
            --config.prior="standard"\
            --config.gpus=1
done