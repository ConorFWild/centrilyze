from pathlib import Path
import json

import fire
import numpy as np

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

def sample_json_from_dir(input_dir_path, output_json_path, random_order=True):
    input_dir_path = Path(input_dir_path).resolve()
    output_json_path = Path(output_json_path).resolve()

    print(f"Input directory is: {input_dir_path}")
    print(f"Output json is: {output_json_path}")

    centrilyze_fs = CentrilyzeDataDir(input_dir_path)

    samples = {}
    for experiment in centrilyze_fs:
        for repeat in experiment:
            for treatment in repeat:
                for embryo in treatment:
                    for image_path in embryo:
                        samples[str(image_path)] = -1

    print(f"Found {len(samples)} samples. Saving to JSON.")

    if random_order:
        new_samples = {}
        keys = list(samples.keys())
        random_reordering = np.random.choice(keys, len(keys), replace=False)
        for key in random_reordering:
            new_samples[key] = samples[key]

        with open(output_json_path, "w") as f:
            json.dump(new_samples, f)

    else:

        with open(output_json_path, "w") as f:
            json.dump(samples, f)


if __name__ == "__main__":
    fire.Fire(sample_json_from_dir)