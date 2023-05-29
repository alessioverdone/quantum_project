import os
import csv
import glob
import random
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

class Dataset(data.Dataset):
    # mapping table of label and index


    def __init__(self, train, **kwargs):
        super(Dataset, self).__init__()

        self.str2label = {"buildings": 0, "forest": 1, "glacier": 2, "mountain": 3, "sea": 4, "street": 5}
        self.label2str = {0: "buildings", 1: "forest", 2: "glacier", 3: "mountain", 4: "sea", 5: "street"}

        self.data = list()
        self.size = kwargs.get("size", None)
        self.data_root = kwargs.get("data_root", "./dataset")
        # self.data_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

        # load csv file and import file paths
        main_dir = "seg_train" if train else "seg_test"
        print(os.path.join(self.data_root, main_dir))

        for current_dir in os.listdir(os.path.join(self.data_root,main_dir)):
            for current_file in os.listdir(os.path.join(self.data_root,main_dir,current_dir)):
                path = os.path.join(self.data_root,main_dir,current_dir,current_file)
                self.data.append((path,current_dir))
                print(f'Loaded: {path}')

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        path, label = self.data[index]
        image = Image.open(path)

        # resize input images
        if self.size:
            image = image.resize((self.size, self.size), Image.BICUBIC)

        label = self.str2label[label]

        return self.transform(image), label

    def __len__(self):
        return len(self.data)