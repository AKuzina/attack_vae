import wandb
import os
from ml_collections import ConfigDict

from vae.model.vae import StandardVAE
from datasets import load_dataset

USER = ''# wandb user name (will be used to download pre-train models)
PROJECT = 'adv_vae'
api = wandb.Api()


def load_model(idx):
    chpt = os.path.join(PROJECT, idx, 'checkpoints/last.ckpt')

    # load model from wandb
    best_model = wandb.restore(chpt, run_path=os.path.join(USER, PROJECT, idx),
                               replace=True, root='wandb')
    # load model from the saved chackpoint
    vae = StandardVAE.load_from_checkpoint(checkpoint_path=best_model.name)
    return vae


def load_data(idx, mode='test'):
    run = api.run(os.path.join(USER, PROJECT, idx))
    data_module, _ = load_dataset(ConfigDict(run.config))
    data_module.prepare_data()
    if mode == 'test':
        data_module.setup('test')
        return data_module.test_dataloader()
    elif mode == 'train':
        data_module.setup('fit')
        return data_module.train_dataloader()


def check_run(run, name, config):
    if name is None or run.name == name:
        if config is None:
            return True
        for k, v in zip(config, config.values()):
            if not (k in run.config.keys()):
                return False
            if run.config[k] != v:
                return False
        return True


def get_experiments(name=None, config=None):
    """
    Return ids of all the expriments based on the name and/or configuration
    """
    runs = api.runs(os.path.join(USER, PROJECT))
    run_list = []
    for run in runs:
        if check_run(run, name, config):
            run_list.append(run.id)
    if len(run_list) != 1:
        print("Found %i" % len(run_list))
    return run_list


