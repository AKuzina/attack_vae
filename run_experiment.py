import os
import sys
import copy
import pytorch_lightning as pl

from datasets import load_dataset
from vae.model.vae import StandardVAE


# args
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", default="config.py:train")


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

    if args.prior in ['standard', 'realnvp']:
        vae = StandardVAE
    else:
        raise ValueError('Unknown prior type')

    # ------------
    # data
    # ------------
    data_module, args = load_dataset(args)

    # ------------
    # model
    # ------------
    model = vae(args)

    # ------------
    # training
    # ------------
    checkpnts = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_last=True,
    )

    early_stop = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=args.lr_patience,
        verbose=True,
        mode='min',
        strict=False
    )

    # ------------
    # weight and bias + trainer
    # ------------
    os.environ["WANDB_API_KEY"] = ' ' # WAND API KEY HERE
    tags = [
        args.prior,
        args.dataset_name
    ]

    wandb_logger = pl.loggers.WandbLogger(project='adv_vae',
                                          tags=tags,
                                          config=copy.deepcopy(dict(args)),
                                          log_model=True,
                                          entity="", # USER NAME HERE
                                          )

    trainer = pl.Trainer(gpus=args.gpus,
                         max_epochs=args.max_epochs,
                         callbacks=[early_stop],
                         logger=wandb_logger,
                         checkpoint_callback=checkpnts  # in newer lightning this goes to callbaks as well
                         )

    trainer.fit(model, datamodule=data_module)

    # ------------
    # testing
    # ------------
    result = trainer.test(datamodule=data_module)
    print(result)


if __name__ == "__main__":
    app.run(cli_main)