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

from centrilyze import constants


def label_to_tensor(label, classes):
    index = classes[label]

    return index


image_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.Normalize(0.0, 1.0, inplace=False),
        torchvision.transforms.RandomAffine(
            (-180.0, 180.0),
            translate=(0.25, 0.25),
            interpolation=transforms.functional.InterpolationMode.BILINEAR,
        ),
        torchvision.transforms.GaussianBlur(
            (3, 3),
            sigma=(0.01, 2.0),
        ),
    ]
)


def target_transform(x):
    return label_to_tensor(x, constants.classes)
