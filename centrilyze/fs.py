import re

def path_to_experiment_particle_frame(path):
    regex = "_particle\[([0-9]+)\]_frame\[([0-9]+)\]"
    capture = re.findall(regex, path)
    experiment = str(capture[0][1])
    particle = int(capture[0][1])
    frame = int(capture[0][2])
    return experiment, particle, frame

def path_to_annotation_experiment_particle_frame(path):
    regex = "(*+)_particle\[([0-9]+)\]_frame\[([0-9]+)\]"
    capture = re.findall(regex, path)
    annotation = path.parts[-2]
    experiment = str(capture[0][1])
    particle = int(capture[0][1])
    frame = int(capture[0][2])
    return annotation, experiment, particle, frame


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