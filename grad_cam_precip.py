import torch
from utils import model_classes
import argparse
from utils.dataset_precip import precipitation_maps_oversampled_h5
import matplotlib.pyplot as plt
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image, scale_cam_image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from tqdm import tqdm
import numpy as np
from root import ROOT_DIR
from models.layers import DepthwiseSeparableConv
from models.unet_parts_depthwise_separable import DoubleConvDS
import pathlib


def get_data(file, in_channels):
    dataset = precipitation_maps_oversampled_h5(file, in_channels, 1, train=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True) 
    return dataloader

def run_cam(model, target_layers, device, test_dataloader, target_names):
    test_dl = test_dataloader
    count = 0
    for x, y_true in tqdm(test_dl, leave=False):
        count += 1
        if count % 99 == 0:
            x = x.to(torch.device(device))
            output = model(x)
            mask = np.digitize((output[0][0] * 47.83 * 12).detach().cpu().numpy(), np.array([1.5]), right=True) 
            mask_float = np.float32(mask)
            image = torch.stack([x[0][-1], x[0][-1], x[0][-1]], dim=2)
            image = (image - image.min()) / (image.max() - image.min())
            mean = torch.mean(image)
            image = image.cpu().numpy()
            mean = np.mean(image)
            image[image[:,:,0] > mean, 0] =  102 / 255 
            image[image[:,:,1] > mean,1] =  51 / 255
            image[:,:,2] = 0
            targets = [SemanticSegmentationTarget(0, mask_float)]
            use_cuda = (device == 'cuda')
            cam_image = []
            for layer in target_layers:
                with GradCAM(model=model, target_layers=layer, use_cuda=use_cuda) as cam:
                    grayscale_cam = cam(input_tensor=x, targets=targets)[0, :]
                    cam_image.append(show_cam_on_image(image, grayscale_cam, use_rgb=True))
        
        

        
            
            '''conf = {'left': 0.125,
                    'bottom' : 0.11,
                    'right':0.283,
                    'top' : 0.874,
                    'wspace':0,
                    'hspace':0}'''

            fig, ax = plt.subplots(5, 4, figsize=(20, 20), ) #gridspec_kw=conf
            for i in range(5):
                for j in range(4):
                    
                    if j == 0:
                        ax[i,j].set_ylabel(f'Encoder Level {i+1}')

                    ax[i,j].imshow(cam_image[4*i + j])
                    if i == 0 and j == 0:
                        ax[i,j].title.set_text(f'Double Convolution Block')
                    elif i == 0 and j ==1:
                        ax[i,j].title.set_text(f'Shuffled DS Convolution')
                    elif i == 0 and j ==2:
                        ax[i,j].title.set_text(f'DS Convolution')
                    elif i == 0 and j ==3:
                        ax[i,j].title.set_text(f'Shuffle Attention')
                    #ax[i,j].axis('off')

            fig, ax = plt.subplots(4,2)
            for i in range(4):
                for j in range(2):
                    if j == 0:
                         ax[i,j].set_ylabel(f'Decoder Level {i+1}')
                    ax[i,j].imshow(cam_image[2*i + j + 20])
                    if i == 0 and j == 0:
                        ax[i,j].title.set_text(f'Double DS Convolution')
                    elif i == 0 and j ==1:
                        ax[i,j].title.set_text(f'UpSampler')

            plt.show()

            

def get_layers(model, idx, layer):
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, layer):
            layers.append(module)
    return layers[idx]

if __name__ == '__main__': 

    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    model_name = 'UNetDSShuffle_Attention4RedV26O' #UNetDSShuffle_Attention3RedV212O UNetDS_Attention12
    model_file = 'C:\\Users\\marco\\OneDrive\\Desktop\\Master\\Code\\SmaAt-UNet\\lightning\\precip_regression\\comparison\\1_Outputs\\UNetDSShuffle_Attention4RedV26O_rain_threshhold_50_epoch=55-val_loss=0.149119.ckpt'
    dataset_folder = ROOT_DIR / "data" / "precipitation" / "train_test_2016-2019_input-length_12_img-ahead_6_rain-threshhold_50.h5"
    device = 'cuda'

    model, name = model_classes.get_model_class(model_name)
    model = model.load_from_checkpoint(model_file)

    dataloader = get_data(dataset_folder, 12)

    print(model.cbam5.sweight)

    target_layers = [ [model.inc],[get_layers(model.inc, 0, DepthwiseSeparableConv)],[get_layers(model.inc, 1, DepthwiseSeparableConv)],[model.cbam1],
                      [model.down1],[get_layers(model.down1, 0, DepthwiseSeparableConv)],[get_layers(model.down1, 1,DepthwiseSeparableConv)],[model.cbam2], 
                      [model.down2],[get_layers(model.down2, 0, DepthwiseSeparableConv)],[get_layers(model.down2, 1, DepthwiseSeparableConv)],[model.cbam3], 
                      [model.down3],[get_layers(model.down3, 0, DepthwiseSeparableConv)],[get_layers(model.down3, 1, DepthwiseSeparableConv)],[model.cbam4], 
                      [model.down4],[get_layers(model.down4, 0, DepthwiseSeparableConv)],[get_layers(model.down4, 1, DepthwiseSeparableConv)],[model.cbam5],
                      [model.up1], [model.up1.up],[model.up2], [model.up2.up], [model.up3], [model.up3.up], [model.up4], [model.up4.up]] 
    target_names = ['cbam1','cbam2', 'cbam3', 'cbam4', 'cbam5',]

    run_cam(model, target_layers, device, dataloader, target_names)



