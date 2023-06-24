from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import sys, yaml, os

class MyDataset(Dataset):
    def __init__(self, data_path,trans):

        self.data = []
        self.trans = trans
        imglist = os.listdir(data_path)
        for img in imglist:
            self.data.append(os.path.join(data_path, img))

        print('len(data) is ', len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]

        img = Image.open(img_path)
        img = np.asarray(img).astype('uint8')

        if len(img.shape) == 2:
            img = np.concatenate(3 * [img[..., None]], axis=2)
        if img.shape[2] != 3:
            img = img[:, :, :3]

        img = Image.fromarray(img)
        
        img = self.trans(img)

        return img



if __name__ == '__main__':
    pass
