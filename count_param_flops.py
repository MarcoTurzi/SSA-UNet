from models import unet_precip_regression_lightning as unet
from fvcore.nn import FlopCountAnalysis
from pathlib import Path
import pathlib
import torch
import time
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd

def get_class(model_name, path):

    if 'SmaAt-UNet' in model_name:

        model = unet.UNetDS_Attention12
        model_path = path.glob('UNetDS_Attention*')


    elif 'SSA-UNet12v2' in model_name:

        model = unet.UNetDSShuffle_Attention4RedV26O
        model_path = path.glob('UNetDSShuffle_Attention4RedV26O*')

    elif 'SSA-UNet' in model_name:

        model = unet.UNetDSShuffle_Attention3RedV212O
        model_path = path.glob('UNetDSShuffle_Attention3RedV212O*')

    else:

        model = unet.UNetDS
        model_path = path.glob('UNetDS12*')
    
    model_path = list(model_path)

    return model, model_path[0]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

if __name__ == '__main__':

    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

    models = [ 'SmaAt-UNet12','SSA-UNet12', 'SSA-UNet12v2']

    folder = Path('C:\\Users\\marco\\OneDrive\\Desktop\\Master\\Code\\SmaAt-UNet\\lightning\\precip_regression\\comparison\\12_Outputs')

    model_metrics = {}


    for model_name in models:

        model, model_path = get_class(model_name, folder)
        model = model.load_from_checkpoint(model_path)
        model.eval()

        image = torch.rand(size=(6,12, 288, 288)).to('cuda')
        
        flops = FlopCountAnalysis(model, image,)
        model_metrics[model_name] = {}
        model_metrics[model_name]['flops'] = flops.total()

        model_metrics[model_name]['parameters'] = count_parameters(model)
        
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        print(torch.cuda.is_available())
        with torch.no_grad():
            final_time = 0
            for i in range(1):
                start.record()
                model(image)
                end.record()
                torch.cuda.synchronize()
                final_time += start.elapsed_time(end)
        
        model_metrics[model_name]['inference_time'] = final_time / 1

    

    for k, v in model_metrics.items():
        print(f"Flops {k}: {v['flops']}")
        print(f"Parameters {k}: {v['parameters']}")
        print(f"Inference Time {k}: {v['inference_time']}")

    # Create a subplot grid with 2 rows and 2 columns
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    # Merge the second row's columns to create a single plot
    ax[1, 1].remove()
    ax[1, 0] = fig.add_subplot(2, 1, 2)

    # Prepare data for FLOPS plot
    data_flops = {
        'Model': list(model_metrics.keys()),
        'FLOPS': [v['flops'] for v in model_metrics.values()]
    }
    df_flops = pd.DataFrame(data_flops)

    # Create scatter plot for FLOPS
    sn.scatterplot(data=df_flops, x='Model', y='FLOPS', style='Model', hue='Model', markers=['o', 's', 'D'], ax=ax[0, 0], legend=False)
    ax[0, 0].set_ylabel("FLOPS")


    # Prepare data for Parameters plot
    data_params = {
        'Model': list(model_metrics.keys()),
        'Parameters': [v['parameters'] for v in model_metrics.values()]
    }
    df_params = pd.DataFrame(data_params)

    # Create scatter plot for Parameters
    sn.scatterplot(data=df_params, x='Model', y='Parameters', style='Model', hue='Model', markers=['o', 's', 'D'], ax=ax[0, 1], legend=False)
    ax[0, 1].set_ylabel("Parameters")

    # Prepare data for Inference Time plot
    data_inference = {
        'Model': list(model_metrics.keys()),
        'inference_time': [v['inference_time'] for v in model_metrics.values()]
    }
    df_inference = pd.DataFrame(data_inference)

    # Create scatter plot for Inference Time
    sn.scatterplot(data=df_inference, x='Model', y='inference_time', style='Model', hue='Model', markers=['o', 's', 'D'], ax=ax[1, 0], legend=False)
    ax[1, 0].set_ylabel("Inference Time")

    plt.tight_layout()
    plt.show()