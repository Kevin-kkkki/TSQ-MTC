import os

import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms


data_transforms = {
    'train':
        transforms.Compose([
        transforms.Resize(224),
        transforms.RandomRotation(45), 
        transforms.CenterCrop(224), 
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5), 
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ]),
    'valid':
        transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class CustomDataset(Dataset):
    def __init__(self, root_dir, datatype, transform=None):
        self.root_dir = root_dir
        # self.classes = sorted(os.listdir(root_dir))
        if datatype == 'SAR':
            self.classes = [
                'amphibious_militaryships_SAR',
                'bulkcargo_civilianships_SAR',
                'airliner_civilianaircrafts_SAR',
                'tanker_civilianships_SAR',
                'aircraftcarrier_militaryships_SAR',
                'transportplane_militaryaircrafts_SAR',
                'containership_civilianships_SAR',
                'DFC_militaryships_SAR',
            ]
        elif datatype == 'RGB':
            self.classes = [
                'amphibious_militaryships_RGB',
                'bulkcargo_civilianships_RGB',
                'airliner_civilianaircrafts_RGB',
                'tanker_civilianships_RGB',
                'aircraftcarrier_militaryships_RGB',
                'transportplane_militaryaircrafts_RGB',
                'containership_civilianships_RGB',
                'DFC_militaryships_RGB',
            ]
        self.data = []
        self.categories = []
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.ToTensor(),
        ]) if transform is None else transform
        self.sar_flag = 0
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                # split name
                category_label, main_category_label, acquisition_label = class_name.split('_')
                if acquisition_label != datatype:
                    continue
                if acquisition_label == "SAR":
                    category_label = category_label + "_SAR"
                    self.sar_flag = 1
                self.categories.append(category_label)
                for image_name in sorted(os.listdir(class_path)):
                    if image_name.endswith('.jpg') or image_name.endswith('.png') or image_name.endswith('.JPG'):
                        image_path = os.path.join(class_path, image_name)
                        self.data.append((image_path, category_label, main_category_label, acquisition_label))
        print(self.categories)
        # self.categories_to_index = {class_name: idx for idx, class_name in enumerate(self.categories)}
        if self.sar_flag == 0:
            self.categories_to_index = {
                "amphibious": 0,
                "bulkcargo": 1,
                "airliner": 2,
                "tanker": 3,
                "aircraftcarrier": 4,
                "transportplane": 5,
                "containership": 6,
                "DFC": 7,
            }
        else:
            self.categories_to_index = {
                "amphibious_SAR": 0,
                "bulkcargo_SAR": 1,
                "airliner_SAR": 2,
                "tanker_SAR": 3,
                "aircraftcarrier_SAR": 4,
                "transportplane_SAR": 5,
                "containership_SAR": 6,
                "DFC_SAR": 7,
            }
        print(self.categories_to_index)
        self.main_categories_to_index = {
            "militaryships": 0,
            "civilianships": 1,
            "militaryaircrafts": 2,
            "civilianaircrafts": 3
        }
        self.acquisition_to_index = {
            "RGB": 0,
            "SAR": 1
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, category_label, main_category_label, acquisition_label = self.data[idx]
        image = Image.open(image_path)
        image = self.transform(image)
        category_label = torch.tensor(self.categories_to_index[category_label], dtype=torch.long)
        main_category_label = torch.tensor(self.main_categories_to_index[main_category_label], dtype=torch.long)
        acquisition_label = torch.tensor(self.acquisition_to_index[acquisition_label], dtype=torch.long)
        return image, category_label, main_category_label, acquisition_label
