import os
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

# Centrilyze imports
from centrilyze import (CentrioleImageFiles,
                        ImageDataset,
                        CentrioleImageModel,
                        CentrilyzeDataDir,
                        HMM,
                        constants,
                        image_transform,
                        target_transform,
                        annotate,
                        nest_annotation_keys,
                        get_sequence_matrix,
                        get_transition_count_matrix,
                        get_transition_rate_matrix,
                        get_confusion_matrix,
                        reannotate,
                        save_all_state_figs,
                        )


def load_embryo_dataset(embryo, batch_size):
    centriole_image_files = CentrioleImageFiles.from_unannotated_images(embryo.path)
    image_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.Normalize(0.0, 1.0, inplace=False),
        ]
    )
    testset = ImageDataset.from_centriole_image_files(
        centriole_image_files,
        image_transform,
        target_transform,
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False
                                             )

    return testloader


def write_centrilyze_results_to_excel(experiment_results, out_dir):
    for experiment_name, experiment_result in experiment_results.items():
        experiment_out_dir = out_dir / experiment_name
        if not experiment_out_dir.exists():
            os.mkdir(experiment_out_dir)

        workbook_file = experiment_out_dir / "experiment.xlsx"

        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(workbook_file)

        for embryo_name, annotations in experiment_result.items():

            records = []
            for annotation_key, annotation in annotations.items():
                experiment, particle, frame = annotation_key
                annotation = annotation["assigned"]
                record = {
                    "Particle": particle,
                    "Frame": frame,
                    "Annotation": annotation,
                }

            df = pd.DataFrame(records)
            df_sorted = df.sort(['Particle', 'Frame'], ascending=[True, True])

            # Convert the dataframe to an XlsxWriter Excel object.
            df_sorted.to_excel(writer, sheet_name='embryo_name')

            # Close the Pandas Excel writer and output the Excel file.
            writer.save()


# Test function
def centrilyze_test(
        test_data_dir="/nic/data/test/col_02-20220318T142333Z-001",
        model_dir="/nic/models/model_resnet_18_high_low_affine_149.pyt",
        output_dir="/nic/output/col_02",
        n_iter=1000,
        batch_size=4,
):
    # Settings
    test_data_dir = Path(test_data_dir).resolve()
    model_dir = Path(model_dir).resolve()
    # annotations_file = Path("/nic/annotations.json")
    # sequences_file = Path("/nic/sequences.npy")
    # emission_matrix_path = Path("/nic/emission_matrix.npy")
    # emission_matrix_path_three_classes = Path("/nic/emission_matrix_three_classes.npy")
    output_dir = Path(output_dir).resolve()

    # Get the model
    model = CentrioleImageModel()

    # Load the trained model params
    image_model = CentrioleImageModel()
    image_model.load_state_dict(model_file, map_location=torch.device("cpu"))

    # Get the test data
    centrilyze_test_data = CentrilyzeDataDir(test_data_dir)

    # For each experiment
    experiment_results = {}
    for experiment in centrilyze_test_data:
        # For each embryo
        embryo_results = {}
        for embryo in experiment:
            # Load the Embryo Data
            testloader = load_embryo_dataset(embryo, batch_size)

            # Annotate the data
            annotations = annotate(image_model, testloader)

            # Output the Confusion Matrix
            # confusion_matrix_test_table = get_confusion_matrix(annotations)

            #
            embryo_results[embryo.name] = annotations

        experiment_results[experiment.name] = embryo_results

    # Write to excel
    write_centrilyze_results_to_excel(experiment_results)


# main
if __name__ == "__main__":
    fire.Fire(centrilyze_test)
