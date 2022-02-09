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


def get_confusion_matrix(annotations):
    confusion_matrix_test = np.zeros((len(constants.classes), len(constants.classes)))

    for key, annotation in annotations.items():
        confusion_matrix_test[annotation["true"], annotation["assigned"]] += 1

    confusion_matrix_test_table = pd.DataFrame(
        data=confusion_matrix_test,
        index=list(constants.classes.keys()),
        columns=list(constants.classes.keys()),
    )

    return confusion_matrix_test_table