import os
from pathlib import Path
import json
import re

import fire

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

from loguru import logger

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

    return testset, testloader


def write_centrilyze_results_to_excel(experiment_results, out_dir):
    if not out_dir.exists():
        os.mkdir(out_dir)

    # Sample annotations
    for experiment_name, experiment_result in experiment_results.items():
        experiment_out_dir = out_dir / experiment_name
        if not experiment_out_dir.exists():
            os.mkdir(experiment_out_dir)

        workbook_file = experiment_out_dir / f"{experiment_name}_sample_annotations.xlsx"

        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(workbook_file)
        records = []

        for repeat_name, repeat_result in experiment_result.items():

            for treatment_name, treatment_result in repeat_result.items():

                for embryo_name, annotations in treatment_result.items():

                    for annotation_key, annotation in annotations.items():
                        experiment, particle, frame = annotation_key
                        annotation = annotation["assigned"]
                        record = {
                            "Repeat": repeat_name,
                            "Treatment": treatment_name,
                            "Embryo": embryo_name,
                            "Particle": particle,
                            "Frame": frame,
                            "Annotation": annotation,
                        }
                        records.append(record)

        df = pd.DataFrame(records)
        df_sorted = df.sort_values(["Repeat", "Treatment", "Embryo", 'Particle', 'Frame'],
                                   ascending=[True, True, True, True, True])

        # Convert the dataframe to an XlsxWriter Excel object.
        df_sorted.to_excel(writer, sheet_name=experiment_name, index=False)

        # Close the Pandas Excel writer and output the Excel file.
        writer.save()

    # Summaries
    for experiment_name, experiment_result in experiment_results.items():
        experiment_out_dir = out_dir / experiment_name
        if not experiment_out_dir.exists():
            os.mkdir(experiment_out_dir)

        workbook_file = experiment_out_dir / f"{experiment_name}_summary.xlsx"

        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(workbook_file)

        records = []
        for repeat_name, repeat_result in experiment_result.items():

            for treatment_name, treatment_result in repeat_result.items():

                for embryo_name, annotations in treatment_result.items():

                    count_dict = {}

                    for annotation_class in constants.classes:
                        count_dict[annotation_class] = 0

                    for annotation_key, annotation in annotations.items():
                        # experiment, particle, frame = annotation_key
                        annotation = annotation["assigned"]
                        count_dict[constants.classes_inverse[annotation]] += 1

                    record = {
                        "Repeat": repeat_name,
                        "Treatment": treatment_name,
                        "Embryo": embryo_name,
                    }
                    for annotation_class in ["Oriented", "Precieved_Oriented", "Slanted", "Precieved_Not_Oriented",
                                             "Not_Oriented", "Unidentified", "No_sample"]:
                        annotation_count = count_dict[annotation_class]
                        if len(annotations) == 0:
                            record[annotation_class] = 0.0
                        else:
                            record[annotation_class] = annotation_count / len(annotations)
                    records.append(record)

        df = pd.DataFrame(records)
        df_sorted = df.sort_values(["Repeat", "Treatment", "Embryo"],
                                   ascending=[True, True, True])

        # Convert the dataframe to an XlsxWriter Excel object.
        df_sorted.to_excel(writer, sheet_name=experiment_name, index=False)

        # Close the Pandas Excel writer and output the Excel file.
        writer.save()


def save_annotation_figures(annotations, testset, output_dir):
    # Get nested annotations
    nested_annotations = nest_annotation_keys(annotations)

    # Reannotate
    reannotations = reannotate(nested_annotations, constants.annotation_mapping)

    # Get the states
    states = {}
    for experiment, particles in reannotations.items():
        for particle, frames in particles.items():
            frame_list = []
            for frame, annotation in frames.items():
                frame_list.append(annotation["assigned"])
                frame_array = np.array(frame_list)
            if tuple(np.unique(frame_array)) not in states:
                states[tuple(np.unique(frame_array))] = set()
            states[tuple(np.unique(frame_array))] = states[tuple(np.unique(frame_array))].union(
                ((experiment, particle),))

    # Save
    save_all_state_figs(
        states,
        testset,
        output_dir,
        reannotations,
    )


class FSModel:
    def __init__(self,
                 test_data_dir,
                 model_dir,
                 output_dir
                 ):
        self.test_data_dir = Path(test_data_dir).resolve()
        self.model_dir = Path(model_dir).resolve()
        self.output_dir = Path(output_dir).resolve()

        if not self.output_dir.exists():
            os.mkdir(self.output_dir)

        self.centrilyze_test_data = CentrilyzeDataDir(self.test_data_dir)


def make_experiment_dir(experiment, fs):
    experiment_images_dir = fs.output_dir / f"{experiment.name}_annotated_particles"
    if not experiment_images_dir.exists():
        os.mkdir(experiment_images_dir)


def annotate_embryo(experiment, embryo, batch_size, image_model, fs):
    logger.debug(f"Annotating embryo: {embryo.name}...")

    # Load the Embryo Data
    testset, testloader = load_embryo_dataset(embryo, batch_size)

    # Annotate the data
    annotations = annotate(
        image_model,
        testloader,
    )
    # Save annotation plots
    experiment_images_dir = fs.output_dir / f"{experiment.name}_annotated_particles"

    save_annotation_figures(
        annotations,
        testset,
        experiment_images_dir
    )

    #
    return annotations


def annotate_fs(fs: FSModel, batch_size, image_model):
    fs.centrilyze_test_data.map_experiments(lambda x: make_experiment_dir(x, fs))

    experiment_results = fs.centrilyze_test_data.map_embryos(
        lambda experiment, repeat, treatment, embryo: annotate_embryo(experiment, embryo, batch_size, image_model, fs)
    )

    return experiment_results

# Test function
def centrilyze_test(
        test_data_dir=r"C:\nic\test_data",
        # test_data_dir=r"C:\nic\data_for_conor\data_for_conor",
        model_dir=r"C:\nic\new_test_script_test_folder\model",
        output_dir=r"/nic/new_test_script_test_folder_2",
        n_iter=1000,
        batch_size=4,
):
    # Settings
    logger.info("Finding test data...")
    fs = FSModel(
        test_data_dir=test_data_dir,
        model_dir=model_dir,
        output_dir=output_dir,
    )

    # Load the trained model params
    logger.info("Loading model parameters...")
    image_model = CentrioleImageModel()
    image_model.load_state_dict(
        fs.model_dir / "model.pyt",
        map_location=torch.device("cpu"),
    )

    # Annotate the embryos For each experiment
    logger.info("Annotating embryos...")
    experiment_results = annotate_fs(fs, batch_size, image_model)

    # Write to excel
    logger.info(f"Saving results to {output_dir}...")
    write_centrilyze_results_to_excel(experiment_results, fs.output_dir)

    # Done
    logger.info("Done!")


# main
if __name__ == "__main__":
    fire.Fire(centrilyze_test)
