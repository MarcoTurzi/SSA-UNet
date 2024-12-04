import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision
from pathlib import Path
from torchvision.transforms import v2
import matplotlib.pyplot as plt

class Cloud_Dataset(Dataset):

    def __init__(self, folder, train=True, num_input=4, num_output=6, transform=None, target_transform=None):
        

        prefix = 'train' if train else 'test'
        self.dataset_files = list(Path(folder).glob(f"{prefix}/*.npz"))
        self.num_input = num_input
        self.num_output = num_output
        self.transform  = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.dataset_files)

    def __getitem__(self, index):
        
        image = np.load(self.dataset_files[index],)['arr_0']
        

        item = self.transform(image)

        #item = image.transpose(2, 0, 1)
        
        input = item[:self.num_input]
        
        label = item[self.num_input:]
        
        return input, label


if __name__ == '__main__':

    transform = v2.Compose([v2.ToImage(),
                            v2.ToDtype(torch.float32)])

    dataset = Cloud_Dataset(folder='data/cloud', transform=transform, target_transform=transform)
    item, label = dataset.__getitem__(1)
    fig, ax = plt.subplots(1,4)
    for i in range(4):
        ax[i].imshow(item[i,:,:])
    
    fig, ax = plt.subplots(1,6)
    for i in range(6):
        ax[i].imshow(label[i])

    plt.show()

    print(item.dtype, label.dtype)

        
     
