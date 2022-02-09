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


def annotate(image_model, dataloader):
    # confusion_matrix_test = np.zeros((len(constants.classes), len(constants.classes)))
    annotations = {}

    model = image_model.model
    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]

        key = data["key"]
        # print(key)
        path = data["path"]
        inputs = data["image"]
        labels = data["label"]

        # forward + backward + optimize
        outputs = model(inputs)

        for j in [0, 1, 2, 3]:
            # confusion_matrix_test[labels[j], torch.argmax(outputs[j])] += 1
            true_label = int(labels[j])
            assigned_label = int(torch.argmax(outputs[j]).detach().cpu())
            experiment = str(key[0][j])
            particle = int(key[1][j])
            frame = int(key[2][j])

            annotations[(experiment, particle, frame)] = {"true": true_label, "assigned": assigned_label}

    return annotations