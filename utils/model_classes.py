from models import unet_precip_regression_lightning as unet_regr
import lightning.pytorch as pl
from typing import Tuple, Type


def get_model_class(model_file) -> Tuple[Type[pl.LightningModule], str]:
    # This is for some nice plotting
    if "UNetDSShuffle_Attention3Red" in model_file:
        model_name = "SSA_UNet"
        model = unet_regr.UNetDSShuffle_Attention3RedV212O
    elif "UNetDS_Attention12" in model_file:
        model_name = "SmaAt-UNet"
        model = unet_regr.UNetDS_Attention12
    elif "UNetDS_Attention_4kpl" in model_file:
        model_name = "UNetDS Attention with 4kpl"
        model = unet_regr.UNetDS_Attention
    elif "UNetDSShuffle_Attention4Red" in model_file:
        model_name = "SSA-UNetV2"
        model = unet_regr.UNetDSShuffle_Attention4RedV26O
    elif "UNetDS_Attention_4CBAMs" in model_file:
        model_name = "UNetDS Attention 4CBAMs"
        model = unet_regr.UNetDS_Attention_4CBAMs
    elif "UNetDS_Attention" in model_file:
        model_name = "SmaAt-UNet"
        model = unet_regr.UNetDS_Attention
    elif "UNetDS_DecAttention" in model_file:
        model_name = "UNetDS Decoupled"
        model = unet_regr.UNetDS_DecAttention
    elif "UNetDS" in model_file:
        model_name = "UNetDS"
        model = unet_regr.UNetDS
    elif "UNet" in model_file:
        model_name = "UNet"
        model = unet_regr.UNet
    else:
        raise NotImplementedError("Model not found")
    return model, model_name
