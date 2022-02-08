from pathlib import Path
import json
import re

import torch
from torch import Tensor
import torch.optim as optim
from torch.utils.data import Dataset
from torch import nn
from torch import functional as F

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import image as mpl_image
from typing import Type, Any, Callable, Union, List, Optional


class ImageDataset(Dataset):
    def __init__(self, data, image_transform, target_transform):
        self.data = data
        self.keys = list(self.data.keys())
        self.image_transform = image_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_key = self.keys[idx]
        sample = self.data[sample_key]
        image = self.image_transform(sample["image"][:, :, :3])
        label = self.target_transform(sample["label"])
        #         print(sample)

        return {"image": image, "label": label, "key": sample_key, "path": str(sample["path"])}

    @staticmethod
    def from_annotated_images():
        ...

    @staticmethod
    def from_unannotated_images():
        ...

    @staticmethod
    def from_centriole_image_files(centriole_image_files, image_transform, target_transform):
        data = {}
        for key, image_path_and_annotation in centriole_image_files.images.items():
            image_path = image_path_and_annotation[0]
            annotation = image_path_and_annotation[1]
            image = mpl_image.imread(str(image_path))
            data[key] = {"image": image, "label": annotation, "path": str(image_path)}

        return ImageDataset(data, image_transform, target_transform)

