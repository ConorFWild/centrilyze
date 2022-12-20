import os
from pathlib import Path

import fire
import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
from loguru import logger

# Centrilyze imports
from centrilyze import (CentrioleImageFiles,
                        ImageDataset,
                        CentrioleImageModel,
                        CentrilyzeDataDir,
                        constants,
                        target_transform,
                        annotate,
                        nest_annotation_keys,
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


def make_sample_sheet(experiment_result):
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

    return df_sorted


def make_summary_sheet(experiment_result):
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

    return df_sorted


def make_summary_sheet_three_classes(experiment_result):
    records = []
    for repeat_name, repeat_result in experiment_result.items():
        for treatment_name, treatment_result in repeat_result.items():
            for embryo_name, annotations in treatment_result.items():
                count_dict = {}

                for annotation_class in constants.classes_reduced:
                    count_dict[annotation_class] = 0

                for annotation_key, annotation in annotations.items():
                    # experiment, particle, frame = annotation_key
                    annotation = annotation["assigned"]
                    count_dict[constants.classes_reduced_inverse[constants.annotation_mapping[annotation]]] += 1

                record = {
                    "Repeat": repeat_name,
                    "Treatment": treatment_name,
                    "Embryo": embryo_name,
                }
                for annotation_class in constants.classes_reduced:
                    annotation_count = count_dict[annotation_class]
                    if len(annotations) == 0:
                        record[annotation_class] = 0.0
                    else:
                        record[annotation_class] = annotation_count / len(annotations)
                records.append(record)

    df = pd.DataFrame(records)
    df_sorted = df.sort_values(["Repeat", "Treatment", "Embryo"],
                               ascending=[True, True, True])

    return df_sorted


def make_transition_sheet(experiment_result):
    records = []
    for repeat_name, repeat_result in experiment_result.items():
        for treatment_name, treatment_result in repeat_result.items():
            for embryo_name, annotations in treatment_result.items():
                # Get nested annotations
                nested_annotations = nest_annotation_keys(annotations)

                # Reannotate
                # reannotations = reannotate(nested_annotations, constants.annotation_mapping)

                # Get the states
                states = {j: {k: 0 for k in range(len(constants.classes))} for j in range(len(constants.classes))}
                for experiment, particles in nested_annotations.items():
                    for particle, frames in particles.items():
                        for j in range(20):
                            if j in frames:
                                if j - 1 in frames:
                                    state_current = frames[j]['assigned']
                                    state_previous = frames[j - 1]['assigned']
                                    states[state_previous][state_current] += 1

                record = {
                    "Repeat": repeat_name,
                    "Treatment": treatment_name,
                    "Embryo": embryo_name,
                }
                for annotation_class_index in states:
                    annotation_class_name = constants.classes_inverse[annotation_class_index]
                    annotation_class_transitions = states[annotation_class_index]
                    class_to_class_transitions = annotation_class_transitions[annotation_class_index]
                    total_transitions_observed = sum(
                        [
                            annotation_class_transitions[k]
                            for k
                            in range(len(constants.classes))
                        ]
                    )
                    logger.debug(f"{annotation_class_name}: {total_transitions_observed}")
                    if total_transitions_observed == 0:
                        record[annotation_class_name] = 0.0
                    else:
                        record[annotation_class_name] = class_to_class_transitions / total_transitions_observed
                records.append(record)

    df = pd.DataFrame(records)
    df_sorted = df.sort_values(["Repeat", "Treatment", "Embryo"],
                               ascending=[True, True, True])

    return df_sorted


def make_transition_sheet_three_classes(experiment_result):
    records = []
    for repeat_name, repeat_result in experiment_result.items():
        for treatment_name, treatment_result in repeat_result.items():
            for embryo_name, annotations in treatment_result.items():
                # Get nested annotations
                nested_annotations = nest_annotation_keys(annotations)

                # Reannotate
                reannotations = reannotate(nested_annotations, constants.annotation_mapping)

                # Get the states
                states = {j: {k: 0 for k in range(len(constants.classes_reduced))} for j in range(
                    len(constants.classes_reduced))}
                for experiment, particles in reannotations.items():
                    for particle, frames in particles.items():
                        for j in range(20):
                            if j in frames:
                                if j - 1 in frames:
                                    state_current = frames[j]['assigned']
                                    state_previous = frames[j - 1]['assigned']
                                    states[state_previous][state_current] += 1

                record = {
                    "Repeat": repeat_name,
                    "Treatment": treatment_name,
                    "Embryo": embryo_name,
                }
                for annotation_class_index in states:
                    annotation_class_name = constants.classes_reduced_inverse[annotation_class_index]
                    annotation_class_transitions = states[annotation_class_index]
                    class_to_class_transitions = annotation_class_transitions[annotation_class_index]
                    total_transitions_observed = sum(
                        [
                            annotation_class_transitions[k]
                            for k
                            in range(len(constants.classes_reduced))
                        ]
                    )
                    logger.debug(f"{annotation_class_name}: {total_transitions_observed}")
                    if total_transitions_observed == 0:
                        record[annotation_class_name] = 0.0
                    else:
                        record[annotation_class_name] = class_to_class_transitions / total_transitions_observed
                records.append(record)

    df = pd.DataFrame(records)
    df_sorted = df.sort_values(["Repeat", "Treatment", "Embryo"],
                               ascending=[True, True, True])

    return df_sorted


def make_excel_file(sheet_frames, workbook_file):
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(workbook_file)

    for sheet_name, df_sheet in sheet_frames.items():
        # Convert the dataframe to an XlsxWriter Excel object.
        df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


def write_centrilyze_results_to_excel(
        experiment_results,
        out_dir,
):
    if not out_dir.exists():
        os.mkdir(out_dir)

    # Sample annotations
    logger.info("Creating sample annotation excel sheets...")
    for experiment_name, experiment_result in experiment_results.items():
        experiment_out_dir = out_dir / experiment_name
        if not experiment_out_dir.exists():
            os.mkdir(experiment_out_dir)

        make_excel_file(
            {experiment_name: make_sample_sheet(experiment_result), },
            experiment_out_dir / f"{experiment_name}_sample_annotations.xlsx"
        )

    # Summaries
    logger.info("Creating experiment summaries...")
    for experiment_name, experiment_result in experiment_results.items():
        experiment_out_dir = out_dir / experiment_name
        if not experiment_out_dir.exists():
            os.mkdir(experiment_out_dir)

        make_excel_file(
            {
                f'{experiment_name}_summary': make_summary_sheet(experiment_result),
                f'{experiment_name}_transitions': make_transition_sheet(experiment_result),
                f'{experiment_name}_summary_3_class': make_summary_sheet_three_classes(experiment_result),
                f'{experiment_name}_transitions_3_class': make_transition_sheet_three_classes(experiment_result),
            },
            experiment_out_dir / f"{experiment_name}_summary.xlsx"
        )


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
    if not fs.output_dir.exists():
        os.mkdir(fs.output_dir)

    experiment_output_dir = fs.output_dir / f"{experiment.name}"
    if not experiment_output_dir.exists():
        os.mkdir(experiment_output_dir)
    experiment_images_dir = experiment_output_dir / f"{experiment.name}_annotated_particles"
    if not experiment_images_dir.exists():
        os.mkdir(experiment_images_dir)

    # save_annotation_figures(
    #     annotations,
    #     testset,
    #     experiment_images_dir
    # )

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
        # test_data_dir=r"C:\nic\test_heirarchical_data_high_quality",
        model_dir=r"C:\nic\new_test_script_test_folder\model",
        output_dir=r"/nic/new_test_script_test_folder_high_quality",
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
    logger.info(fs.centrilyze_test_data.to_dict())


    # Load the trained model params
    logger.info("Loading model parameters...")
    image_model = CentrioleImageModel()
    image_model.load_state_dict(
        fs.model_dir / "model.pyt",
        map_location=torch.device("cpu"),
    )

    # Annotate the embryos For each experiment
    logger.info("Annotating embryos...")
    experiment_results = annotate_fs(
        fs,
        batch_size,
        image_model,
    )

    # Write to excel
    logger.info(f"Saving results to {output_dir}...")
    write_centrilyze_results_to_excel(experiment_results, fs.output_dir)
    logger.info(f"Wrote results to: {fs.output_dir}")

    # Done
    logger.info("Done!")


# main
if __name__ == "__main__":
    fire.Fire(centrilyze_test)
