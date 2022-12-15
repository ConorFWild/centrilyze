from pathlib import Path
import json

import fire

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

def sample_json_from_dir(input_dir_path, output_json_path, ):
    input_dir_path = Path(input_dir_path).resolve()
    output_json_path = Path(output_json_path).resolve()

    centrilyze_fs = CentrilyzeDataDir(input_dir_path)

    samples = {}
    for experiment in centrilyze_fs.experiments:
        for repeat in experiment:
            for treatment in repeat:
                for embryo in treatment:
                    for image_path in embryo:
                        samples[str(image_path)] = -1

    with open(output_json_path, "w") as f:
        json.dump(samples, output_json_path)

if __name__ == "__main__":
    fire.Fire()