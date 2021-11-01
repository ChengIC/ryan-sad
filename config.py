from base.torchvision_dataset import *
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from torchvision import transforms
######################### Custom Dataset Example

'''
Example :
    you must define following datasets:
    1. train_dataset
    2. test_dataset
    or using random_split to split dataset
'''

class dataset(Dataset):
    def __init__(self,transform=None):
        super().__init__()
        self.root_path = './Data'
        self.transform = transform
        self.data = list(map(
            lambda x : (os.path.join(self.root_path + "/normal",x),1),
            os.listdir(os.path.join(self.root_path,'normal'))
            )) + list(map(
            lambda x : (os.path.join(self.root_path + "/abnormal",x),0),
            os.listdir(os.path.join(self.root_path,'abnormal'))
        ))
    def __getitem__(self,index):
        img = Image.open(self.data[index][0])
        label = torch.tensor(self.data[index][1])
        if self.transform!=None:
            img = self.transform(img)
        return img,label,label,label

    def __len__(self):
        return len(self.data)


transform = transforms.Compose(
    [
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ]
)

Custom = CustomDataset(
    dataset(transform=transform),
    dataset(transform=transform)
)

#########################################################################################################

Pretrain = False
'''
Implemented Methods = ['sad','kde','isoforest','ocsvm','semi-dgm','ssad']
'''

task = 'sad'

method_params = {
    
}

pretrain_params = {

}

train_params = {
    'n_epochs':3,
    'dataset':Custom,
}

ae_net_name = ''

net_name    = 'cifar10_LeNet'

save_path   = "./test.pth"

