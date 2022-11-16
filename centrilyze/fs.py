from pathlib import Path
import re


def path_to_experiment_particle_frame(path):
    stem = path.stem
    regex = "(.*)_particle\[([0-9]+)\]_frame\[([0-9]+)\]"
    capture = re.findall(regex, str(stem))
    experiment = str(capture[0][0])
    particle = int(capture[0][1])
    frame = int(capture[0][2])
    return experiment, particle, frame


def path_to_annotation_experiment_particle_frame(path):
    stem = path.stem
    regex = "(.*)_particle\[([0-9]+)\]_frame\[([0-9]+)\]"
    capture = re.findall(regex, str(stem))
    annotation = path.parts[-2]
    experiment = str(capture[0][1])
    particle = int(capture[0][1])
    frame = int(capture[0][2])
    return annotation, experiment, particle, frame


class EmbryoDataDir:
    def __init__(self, embryo_data_dir: Path):
        self.path = embryo_data_dir

        self.image_files = self.get_image_files_from_dir(self.path)

    def get_embryos_from_dir(self, embryo_dir):

        embryo_image_files = []
        for embryo_image_file in embryo_dir.glob("*"):
            if not embryo_image_file.is_dir():
                continue
            else:
                embryo_image_files.append(embryo_image_file)

        return embryo_image_files

    def __iter__(self):
        for image_file in self.image_files:
            yield image_file


class ExperimentDataDir:
    def __init__(self, experiment_data_dir: Path):
        self.path = experiment_data_dir
        self.name = experiment_data_dir.name
        self.embryos = self.get_embryos_from_dir(self.path)

    def get_embryos_from_dir(self, embryos_dir):

        embryos = []
        for directory in embryos_dir.glob("*"):
            if not directory.is_dir():
                continue
            else:
                embryo = ExperimentDataDir(directory)
                embryos.append(embryo)

        return embryos

    def __iter__(self):
        for embryo in self.embryos:
            yield embryo


class CentrilyzeDataDir:
    def __init__(self, centrilyze_data_dir: Path):
        self.path = centrilyze_data_dir
        self.name = centrilyze_data_dir.name
        self.experiments = self.get_experiments_from_dir(self.path)

    def get_experiments_from_dir(self, experiments_dir):

        experiments = []
        for directory in experiments_dir.glob("*"):
            if not directory.is_dir():
                continue
            else:
                experiment = ExperimentDataDir(directory)
                experiments.append(experiment)

        return experiments

    def __iter__(self):
        for experiment in self.experiments:
            yield experiment


class CentrioleImageFiles:
    def __init__(self, images):
        self.images = images

    @staticmethod
    def from_unannotated_images(path, regex="*.png"):
        paths = [x for x in path.rglob("*.png")]

        images = {}
        for p in paths:
            experiment, particle, frame = path_to_experiment_particle_frame(p)
            images[(experiment, particle, frame,)] = (p, "Unidentified",)

        return CentrioleImageFiles(images)

    @staticmethod
    def from_annotated_images(path, regex="*.png"):
        paths = [x for x in path.rglob("*.png")]

        images = {}
        for p in paths:
            annotation, experiment, particle, frame = path_to_annotation_experiment_particle_frame(p)
            images[(experiment, particle, frame,)] = (p, annotation)

        return CentrioleImageFiles(images)
