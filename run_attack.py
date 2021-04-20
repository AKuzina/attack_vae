import os
import sys
import wandb
import pytorch_lightning as pl

from datasets import load_dataset
from utils.wandb import get_experiments, load_model
from attack import trainer

# args
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", default="config.py:attack")


def model_name(args):
    n = args.prior + '_' + args.arc_type + '_' + str(args.z_dim)
    return n


def cli_main(_):
    pl.seed_everything(1234)

    if "absl.logging" in sys.modules:
        import absl.logging

        absl.logging.set_verbosity("info")
        absl.logging.set_stderrthreshold("info")
    args = FLAGS.config
    print(args)

    # ------------
    # data
    # ------------
    data_module, args.model = load_dataset(args.model)
    data_module.setup('test')
    dataloader = data_module.test_dataloader()
    print(args)

    # ------------
    # load pretrained model
    # ------------
    ids = get_experiments(config=args.model)
    model = load_model(ids[0]).vae
    model.eval()

    # ------------
    # wandb
    # ------------
    os.environ["WANDB_API_KEY"] = ''# WAND API KEY HERE
    tags = [
        args.model.prior,
        args.model.dataset_name,
        args.attack.type
    ]

    wandb.init(
        project="adv_vae",
        tags=tags,
        entity='' # USER NAME HERE
    )
    wandb.config.update(flags.FLAGS)

    # run attack
    trainer.train(model, dataloader, args)


if __name__ == "__main__":
    app.run(cli_main)