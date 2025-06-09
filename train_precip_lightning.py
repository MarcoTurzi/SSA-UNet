from root import ROOT_DIR
from pathlib import Path
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from lightning.pytorch import loggers
import argparse
from models import unet_precip_regression_lightning as unet_regr
import torch
from lightning.pytorch.tuner import Tuner
from lightning import seed_everything

import pathlib

def get_batch_size_and_lr(hparams):
    if hparams.model == "UNetDS_Attention":
        net = unet_regr.UNetDS_Attention(hparams=hparams)
    elif hparams.model == "UNet_Attention":
        net = unet_regr.UNet_Attention(hparams=hparams)
    elif hparams.model == "UNet":
        net = unet_regr.UNet(hparams=hparams)
    elif hparams.model == "UNetDS":
        net = unet_regr.UNetDS(hparams=hparams)
    else:
        raise NotImplementedError(f"Model '{hparams.model}' not implemented")

    trainer = pl.Trainer(gpus=hparams.gpus)
    tuner = Tuner(trainer)
    tuner.scale_batch_size(net, mode="power", init_val=8)
    new_batch_size = tuner.scale_batch_size(net, mode="binsearch", init_val=8)
    new_lr = tuner.scale_lr(net, mode="binsearch", init_val=1e-6)
    print("New biggest batch_size: ", new_batch_size)
    print("New biggest lr: ", new_lr)
    return new_batch_size, new_lr


def train_regression(hparams):
    if hparams.model == "SSA_UNetRed":
        net = unet_regr.SSA_UNetRed(hparams=hparams)
    elif hparams.model == "SSA_UNet":
        net = unet_regr.SSA_UNet(hparams=hparams)
    else:
        raise NotImplementedError(f"Model '{hparams.model}' not implemented")

    
    default_save_path = ROOT_DIR /  "lightning" / "precip_regression"

    checkpoint_callback = ModelCheckpoint(
        dirpath=default_save_path / net.__class__.__name__,
        filename=net.__class__.__name__ + "_rain_threshhold_50_{epoch}-{val_loss:.6f}",
        save_top_k=-1,
        verbose=False,
        monitor="val_loss",
        mode="min",
    )
    lr_monitor = LearningRateMonitor()
    tb_logger = loggers.TensorBoardLogger(save_dir=default_save_path, name=net.__class__.__name__)

    earlystopping_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=hparams.es_patience,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        fast_dev_run=hparams.fast_dev_run,
        max_epochs=hparams.epochs,
        default_root_dir=default_save_path,
        logger=tb_logger,
        callbacks=[checkpoint_callback, earlystopping_callback, lr_monitor],
        val_check_interval=hparams.val_check_interval,
    )


    trainer.fit(model=net)
    #trainer.fit(model=net, ckpt_path=hparams.resume_from_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    torch.set_float32_matmul_precision('high')
    seed_everything(42, workers=True)
    parser = unet_regr.Precip_regression_base.add_model_specific_args(parser)
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    parser.add_argument(
        "--dataset_folder",
        default=ROOT_DIR / "data" / "precipitation" / "RAD_NL25_RAC_5min_train_test_2016-2019.h5",
        type=str,
    )
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--fast_dev_run", type=bool, default=False)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--val_check_interval", type=float, default=None)

    args = parser.parse_args()
    # args.fast_dev_run = Truep
    args.n_channels = 12
    # args.gpus = 1
    args.model = "UNetShuffle_Attention"
    args.lr_patience = 4
    args.es_patience = 15
    # args.val_check_interval = 0.25
    args.num_output_images = 12
    args.n_classes = 12 
    args.use_oversampled_dataset = True
    args.dataset_folder = (
        ROOT_DIR / "data" / "precipitation" / "train_test_2016-2019_input-length_12_img-ahead_12_rain-threshhold_50.h5"
    )
    args.resume_from_checkpoint = None

    for m, kd, ku in [("SSA_UNetRed", 2, 2), ("SSA_UNet", 3, 2) ]:
        args.model = m
        args.kernels_per_layer = kd
        args.kernels_per_layer_up = ku
        
        train_regression(args)
    #train_regression(args)
    pathlib.PosixPath = temp

    
