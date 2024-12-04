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
    if hparams.model == "UNetDSShuffle_Attention4RedV2Cloud":
        net = unet_regr.UNetDSShuffle_Attention4RedV2Cloud(hparams=hparams)
    elif hparams.model == "UNetDS_AttentionCloud":
        net = unet_regr.UNetDS_AttentionCloud(hparams=hparams)
    elif hparams.model == "UNetDS_Attention12":
        net = unet_regr.UNetDS_Attention12(hparams=hparams)
    elif hparams.model == "UNetDSShuffle_Attention8G":
        net = unet_regr.UNetDSShuffle_Attention8G(hparams=hparams)
    elif hparams.model == "UNetDSShuffle_Attention32FRed":
        net = unet_regr.UNetDSShuffle_Attention32FRed(hparams=hparams)
    elif hparams.model == "UNetDSShuffle_Attention":
        net = unet_regr.UNetDSShuffle_Attention(hparams=hparams)
    elif hparams.model == "UNetDSShuffle_Attention2":
        net = unet_regr.UNetDSShuffle_Attention2(hparams=hparams)
    elif hparams.model == "UNetDSShuffle_Attention3Red":
        net = unet_regr.UNetDSShuffle_Attention3Red(hparams=hparams)
    else:
        raise NotImplementedError(f"Model '{hparams.model}' not implemented")

    #default_save_path = ROOT_DIR /  "lightning" / "precip_regression"
    default_save_path = Path('/home/mturzi/data/volume_2') /  "lightning" / "precip_regression"

    checkpoint_callback = ModelCheckpoint(
        dirpath=default_save_path / f"{net.__class__.__name__}mse",
        filename=net.__class__.__name__ + "_cloud_{epoch}-{val_loss:.6f}",
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
    parser = unet_regr.Cloud_base.add_model_specific_args(parser)
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    parser.add_argument(
        "--dataset_folder",
        default=ROOT_DIR / "data" / "cloud" ,
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
    args.n_channels = 4
    # args.gpus = 1
    args.model = "UNetShuffle_Attention"
    args.lr_patience = 4
    args.es_patience = 15
    # args.val_check_interval = 0.25
    args.kernels_per_layer = 2
    args.use_oversampled_dataset = True
    args.dataset_folder = (
        ROOT_DIR / "data" / "cloud" 
    )
    args.resume_from_checkpoint = 'C:\\home\\mturzi\\data\\volume_2\\lightning\\precip_regression\\UNetDSShuffle_Attention3RedV2Cloud\\UNetDSShuffle_Attention3RedV2Cloud_cloud_epoch=14-val_loss=12975.147461.ckpt'

    for m in ["UNetDSShuffle_Attention4RedV2Cloud", ]:
        args.model = m
        train_regression(args)
    #train_regression(args)
    pathlib.PosixPath = temp

    
