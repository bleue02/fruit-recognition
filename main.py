from loc import root

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import PIL
import shutil
import glob
import os

class ImageModel:
    def __init__(self,num_workers=8):
        self.num_workers = num_workers

        self.train_set, self.test_set = root()
        all_file_paths = []

        classes = [folder for folder in os.listdir(self.train_set) if os.path.isdir(os.path.join(self.train_set, folder))]
        print('Total of suffle data :', classes) # printing of the all folder suffle result.

        for fruit_folder in classes:
            fruit_folder_path = os.path.join(self.train_set, fruit_folder)
            files_in_fruit_folder = [os.path.join(fruit_folder_path, file) for file in os.listdir(fruit_folder_path) if
                                     os.path.isfile(os.path.join(fruit_folder_path, file))]

            all_file_paths.extend(files_in_fruit_folder)

        # Move the creation of class_to_index outside the loop
        n = 0
        # classes: fath의 n종류의 fruit의 폴더를 하나씩 들어가 classses의 변수에 넣음.
        class_to_index = {class_name: index for index, class_name in enumerate(classes)}
        for class_name, index in class_to_index.items():
            print(f"{n}:, {class_name}: {index}")
            n = n + 1

        self.shuffled_file_paths = all_file_paths

class CustomData(torch.utils.data.Dataset):
    def __init__(self, model, mode='train', transform=None):
        self.files = model.shuffled_file_paths
        self.train = model.train_set
        self.mode = mode
        self.transform = transform

    def __len__(self): # __len__(self) : Dataset의 길이를 반환하기 위한 메소드
        return len(self.files)

    #  mode='train'일 경우에는 label을 반환하고, 'train'용이 아닌 경우에는 label을 모르기 때문에 실제 이미지 파일의 경로를 반환
    def __getitem__(self, index):
        img = PIL.Image.open(os.path.join(self.train, self.files[index]))

        if self.transform:
            img = self.transform(img)
        if self.mode == 'train': # mode : 해당 dataset이 train용인지 eval용인지 체크
            # return img, np.array([self.label])
            return torch.tensor(self.label)
        else:
            return img, self.files[index] # ?

    # noinspection PyMethodMayBeStatic
    def preprocessing(self):
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.RandomCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
        ])
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 244)),
            torchvision.transforms.ToTensor(), # [0,255] 범위를 갖는 픽셀값을 [0,1]의 범위로 변환
        ])

        # DataLoder
    # def dataloder(self, batch_size=64, shuffle=True):
    #     train_loder = torch.utils.DataLoder(self.train, batch_size, shuffle)
    #     valid_loader = ''
    #     test_loader = ''

run = ImageModel()
custom_dataset = CustomData(run)
print(custom_dataset)
