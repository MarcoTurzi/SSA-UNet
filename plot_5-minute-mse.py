import torch
from torch import nn, triangular_solve
import matplotlib.pyplot as plt
import numpy as np
import os
from os import path
import pickle
from tqdm import tqdm
import math
from pathlib import Path
from utils.cloud_dataset import Cloud_Dataset
from root import ROOT_DIR
from utils import data_loader_precip, dataset_precip, data_loader_precip
from utils.model_classes import get_model_class
from models import unet_precip_regression_lightning as unet_regr
import pathlib
from torchvision.transforms import v2
import pandas as pd
import seaborn as sn



def get_binary_metrics(model, test_dl, loss="mse", denormalize=False, threshold=0.5, minutes_ahead=0):
    with torch.no_grad():
      mps = torch.device("cuda")
      model.eval()  # or model.freeze()?
      model.to(mps)

      if loss.lower() == "mse":
          loss_func = nn.functional.mse_loss
      elif loss.lower() == "mae":
          loss_func = nn.functional.l1_loss
      factor = 1
      if denormalize:
          factor = 47.83

      threshold = threshold
      epsilon = 1e-6

      total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0 

      loss_denorm = 0.0

      count = 0
      for x, y_true in tqdm(test_dl, leave=False):
          count += 1
          x = x.to(mps)
          y_true = y_true.to(mps).squeeze()[minutes_ahead]
          
          #x = x.squeeze()[-1].unsqueeze(0)

          y_pred = model(x) #torch.cat([x,x,x,x,x,x], dim=0)#
          y_pred = y_pred.squeeze()[minutes_ahead]

          # denormalize
          y_pred_adj = y_pred * factor
          y_true_adj = y_true * factor
          # calculate loss on denormalized data
          loss = loss_func(y_pred_adj, y_true_adj, reduction='sum')
          loss_denorm += loss
          

          y_pred_adj *= 12.0
          y_true_adj *= 12.0

          # convert to masks for comparison
          y_pred_mask = y_pred_adj > threshold
          y_true_mask = y_true_adj > threshold
          
          
          y_pred_mask = y_pred_mask.cpu()
          y_true_mask = y_true_mask.cpu()
          tn, fp, fn, tp = np.bincount(y_true_mask.view(-1) * 2 + y_pred_mask.view(-1), minlength=4)
          total_tp += tp
          total_fp += fp
          total_tn += tn
          total_fn += fn
          torch.cuda.empty_cache()

      mse_image = loss_denorm / len(test_dl)
      mse_pixel = mse_image / torch.numel(y_true)

      print(f"TP: {total_tp}")
      print(f"FP: {total_fp}")
      print(f"TN: {total_tn}")
      print(f"FN: {total_fn}")
      # get metrics
      precision = total_tp / (total_tp + total_fp + epsilon)
      recall = total_tp / (total_tp + total_fn + epsilon)
      accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + epsilon)
      f1 = 2 * precision * recall / (precision + recall + epsilon)
      csi = total_tp / (total_tp + total_fn + total_fp + epsilon)
      far = total_fp / (total_tp + total_fp + epsilon)
      pod = total_tp / (total_tp + total_fn + epsilon)
      hss = (total_tp * total_tn - total_fn * total_fp) / ((total_tp + total_fn) * (total_fn + total_tn) + (total_tp + total_fp) * (total_fp + total_tn) + epsilon)
    return mse_pixel.item(), mse_image.item(), precision, recall, accuracy, f1, csi, far, pod, hss

def print_binary_metrics(model, data_file, threshold=0.5):
    test_dl = data_file
   
    mse_pixel, mse_image, precision, recall, accuracy, f1, csi, far, pod, hss = get_binary_metrics(model, test_dl, loss="mse",
                                                                                        denormalize=True, threshold=threshold)
    print(
        f"MSE (pixel): {mse_pixel}, MSE (image): {mse_image}, precision: {precision}, recall: {recall}, accuracy: {accuracy}, f1: {f1}, csi: {csi}, far: {far}, pod: {pod}, hss: {hss}")
    return [mse_pixel, mse_image, precision, recall, accuracy, f1, csi, far, pod, hss]


def get_model_losses(model_folder, data_file, loss, denormalize):
    # Save it to a dict that can be saved (and plotted)
    test_losses = dict()
    

    models = [m for m in os.listdir(model_folder) if ".ckpt" in m]
    # dataset = dataset_precip.precipitation_maps_masked_h5(
    dataset = dataset_precip.precipitation_maps_oversampled_h5(
        in_file=data_file,
        num_input_images=12,
        num_output_images=12, 
        train=False)
    
    '''transform = v2.Compose([v2.ToImage(),
                                      v2.ToDtype(torch.float32)])

    dataset = Cloud_Dataset(data_file,
                            train=False,
                            transform=transform,
                            target_transform=transform)'''

    test_dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    is_first = True
    # load the models
    conc = []
    count = 0
    for model_file in models:
         
        
        print(models)
        model, model_name = get_model_class(model_file)
        model = model.load_from_checkpoint(f"{model_folder}/{model_file}")
        print(model_name)
        name = model_name

        threshold = 0.5
        for i in range(12):
            metrics = get_binary_metrics(model, test_dl, loss="mse", denormalize=True, threshold=threshold, minutes_ahead = i)
            if i == 0 and count == 0:
                conc = [[model_name, (i+1)*5, metrics[0], metrics[2], metrics[4], metrics[3] ]]
                count += 1
            else:
                conc.append([model_name, (i+1)*5, metrics[0], metrics[2], metrics[4], metrics[3]])
            print(conc)
    arr = np.array(conc)
    df = pd.DataFrame(np.array(conc),
                      columns=['Model', 'Time', 'MSE', 'Precision', 'Accuracy', 'Recall'])

    print(df.dtypes)

    df['MSE'] = df['MSE'].astype(float)
    df['Accuracy'] = df['Accuracy'].astype(float)
    df['Precision'] = df['Precision'].astype(float)
    df['Recall'] = df['Recall'].astype(float)

    fig, ax = plt.subplots(2, 2)

    sn.lineplot(data=df,
                x='Time', y='MSE', hue='Model',  markers=True, style='Model', dashes=False, ax=ax[0][0])
    
    sn.lineplot(data=df,
                x='Time', y='Precision', hue='Model',  markers=True, style='Model', dashes=False, ax=ax[0][1])
    
    sn.lineplot(data=df,
                x='Time', y='Accuracy', hue='Model',  markers=True, style='Model', dashes=False, ax=ax[1][0])
    sn.lineplot(data=df,
                x='Time', y='Recall', hue='Model',  markers=True, style='Model', dashes=False, ax=ax[1][1])
    plt.show()   
    return test_losses

def losses_to_csv(losses, path):
    csv = "threshold, name, mse (pixel), mse (image), precision, recall, accuracy, f1, csi, far, pod, hss\n"
    for loss in losses:
      row = ",".join(str(l) for l in loss)
      csv += row + "\n"

    with open(path,"w+") as f:
      f.write(csv)

    return csv


if __name__ == '__main__':
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    loss = "mse"
    denormalize = True
    # Models that are compared should be in this folder (the ones with the lowest validation error)
    #model_folder = Path('C:\\home\\mturzi\\data\\volume_2\\lightning\\precip_regression\\Shuffle3RedV212O')#ROOT_DIR / "checkpoints" / "comparison" / "EPSA" Shuffle3RedV212O SmaAt122
    model_folder = Path('C:\\Users\\marco\\OneDrive\\Desktop\\Master\\Code\\SmaAt-UNet\\lightning\\precip_regression\\comparison\\12_Outputs')
    #model_folder = Path('C:\\home\\mturzi\\data\\volume_2\\lightning\\precip_regression\\Shuffle_Cloud2')
    data_file = (
        ROOT_DIR / "data" / "precipitation" / "train_test_2016-2019_input-length_12_img-ahead_12_rain-threshhold_20.h5"
        #ROOT_DIR / "data" / "cloud"
    )
    results_folder = ROOT_DIR / "results" / "SmaCC" / "test"

    test_losses = dict()
    test_losses = get_model_losses(model_folder, data_file, loss, denormalize)
    pathlib.PosixPath = temp
    #print(losses_to_csv(test_losses['binary_50'], (results_folder / "res_50.csv")))

