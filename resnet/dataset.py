import torch.utils.data as data
from PIL import Image
import os
import numpy as np


class WeaponDataset(data.Dataset):

    def __init__(self, filepaths, transform, phase):
        self.filePaths = filepaths
        self.transform = transform
        self.phase = phase

    def __getitem__(self, item):
        image, category = self.load_image(item)
        return self.transform(image, self.phase), category

    def __len__(self):
        return len(self.filePaths)

    def load_image(self, image_index):
        imageInfo = self.filePaths[image_index]
        path = imageInfo['filepath']
        category = imageInfo['category']
        return Image.open(path), category
