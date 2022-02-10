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

from hmmlearn import hmm


from centrilyze import constants


def nest_annotation_keys(annotations):
    nested_annotations = {}

    for key, annotation in annotations.items():
        experiment = key[0]
        particle = key[1]
        frame = key[2]

        if not experiment in nested_annotations:
            nested_annotations[experiment] = {}

        if not particle in nested_annotations[experiment]:
            nested_annotations[experiment][particle] = {}

        if not frame in nested_annotations[experiment][particle]:
            nested_annotations[experiment][particle][frame] = annotation

    return nested_annotations


def reannotate(nested_annotations, annotation_mapping):
    new_annotations = {}
    for experiment, particles in nested_annotations.items():
        if experiment not in new_annotations:
            new_annotations[experiment] = {}

        for particle, frames in particles.items():
            if particle not in new_annotations[experiment]:
                new_annotations[experiment][particle] = {}

            for frame, annotation in frames.items():

                new_annotation = {
                    "true": annotation_mapping[annotation["true"]],
                    "assigned": annotation_mapping[annotation["assigned"]],
                }

                if frame not in new_annotations[experiment][particle]:
                    new_annotations[experiment][particle][frame] = new_annotation

    return new_annotations


def get_sequence_matrix(annotations, frame_numbers, num_classes):
    all_sequences = []
    for experiment, particles in annotations.items():
        for particle, frames in particles.items():
            ordered_annotations = []
            for frame_number in frame_numbers:
                if frame_number not in frames:
                    ordered_annotations.append(num_classes-1)
                else:
                    annotation = frames[frame_number]["assigned"]
                    ordered_annotations.append(annotation)

            if len(frame_numbers) == len(ordered_annotations):
                all_sequences.append(ordered_annotations)


    sequence_matrix = np.array(all_sequences)

    return sequence_matrix


#
def get_transition_count_matrix(sequence_matrix, num_classes):

    naive_transition_matrix = np.zeros((num_classes, num_classes))

    for sequence in sequence_matrix:
        prev_state = num_classes - 1
        for state in sequence:
            naive_transition_matrix[prev_state, state] += 1
            prev_state = state

    return naive_transition_matrix


def get_transition_rate_matrix(transition_count_matrix):
    naive_transition_matrix = transition_count_matrix / np.sum(transition_count_matrix, axis=1).reshape((-1, 1))
    return naive_transition_matrix


class HMM:

    def __init__(self, n_iter=1000):
        self.model = hmm.MultinomialHMM(n_components=7, n_iter=n_iter, params="st", init_params="st")

    def fit(self, sequences_array, lengths):
        self.model.fit(sequences_array.reshape(-1, 1), lengths)

    def predict(self, sequences_array, lengths):
        self.model.predict(sequences_array.reshape(-1, 1), lengths)
