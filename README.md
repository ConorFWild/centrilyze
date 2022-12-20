# Centrilyze

A small python package for analysing images of centrioles.

## setup enviroment
```bash
git clone https://github.com/ConorFWild/centrilyze.git
cd centrilyze
conda create -n centrilyze python=3.7
conda activate centrilyze
conda install pytorch torchvision -c pytorch
conda install -c conda-forge numpy pandas scipy scikit-learn jupyter matplotlib hmmlearn fire openpyxl loguru
```

## Using Centrilyze

Using Centrilyze is as simple as correctly organizing the input data, optionally training the model if you do not have a trained one already, and running the test script to annotate your data.

### Setting Up Training Data

If you do not already have a trained model, then you will need to organize data to train one. For this there are two options, either a JSON output by centrilyze-gui, or a images arranged into the following file structure:

```commandline
centrilyze_data
├── oriented
│   ├── embryo_particle_n_frame_m.png
│   │   ...
│   └── embryo_particle_l_frame_k.png
├── unoriented
│   │   ...
├── undefined
│   └── ...
├── slanted
│   └──  ...
├── precieved_unoriented
│   └── ...
└── precieved_oriented 
    └── ...
```

### Training The Model
When your data is formatted as folders then:

```commandline
python /path/to/centrilyze/scripts/train.py /path/to/training/data/folder /path/to/model/output/folder 
```

When you are using a json from centrilyze-gui then:

```commandline
python /path/to/centrilyze/scripts/train.py /path/to/training/data/folder /path/to/model/output/samples_annotated.json 
```

### Setting Up Test Data

Data for Centrilyze should be stored in a directory formatted as follows:

```
centrilyze_data
├── {experiment_name_1}
│   ├── {repeat_name_1}
│   │   ├── {treatment_name_1}
│   │   │   ├── {embryo_name_1}
│   │   │   ├── {embryo_particle_1_frame_1}.png
│   │   │   ...
│   │   │   └── {embryo_particle_n_frame_20}.png
│   ...
│   └── {embryo_name_j}
... 
└── {experiment_name_k}
```

### Testing New Data

Once you have trained the model, or downloaded a trained model, you can then test

```
python /path/to/centrilyze/scripts/test.py /path/to/testing/data/folder /path/to/model/folder /path/to/testing/output/folder 
```

### Interpreting The Output

Centrilyze outputs two types of Excel files: samples and summaries. The sample files contain the annotation for each image in an experiment. In the summary file, the summary sheet contains counts of images in each state for each treatment. The transition sheet in the summary file contains the probability of transition out of the state for each state in each treatment.

Centrilyze outputs an Excel sheet for each embryo in the:

```
centrilyze_output_dir
├── {experiment_name_1}
│   ├── {experiment_name_1_annotated_samples}.xlsx
│   ├── {experiment_name_1_summary}.xlsx
│   ...
│   └── {experiment_name_j_summary}.xlsx
...
└── {experiment_name_k}
```

Each Excel file will contain a sheet with the annotations for each image.

## Get a copy of the model
located at:

```commandline
https://drive.google.com/drive/folders/1uT5toYVPpbr-4aiN7oZeXRdmWMOl9Rqy?usp=sharing
```

in the data folder.
