# Centrilyze

A small python package for analysing images of centrioles.

## setup enviroment
```bash
git clone https://github.com/ConorFWild/centrilyze.git
cd nic
conda create -n nic python=3.7
conda activate nic
conda install pytorch torchvision -c pytorch
conda install -c conda-forge numpy pandas scipy scikit-learn jupyter matplotlib hmmlearn fire openpyxl
jupyter notebook
```

## Using Centrilyze

### Setting Up Data

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

### Training The Model

```
python /path/to/centrilyze/scripts/train.py /path/to/training/data/folder /path/to/model/output/folder 
```

### Testing New Data

Once you have trained the model, or downloaded a trained model, you can then test

```
python /path/to/centrilyze/scripts/test.py /path/to/testing/data/folder /path/to/model/folder /path/to/testing/output/folder 
```

### Interpreting The Output

Centrilyze outputs an excel sheet for each embryo in the:

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

## Open notebook
open notebooks
open test.ipynb

## Get a copy of the model
located at:

https://drive.google.com/drive/folders/1uT5toYVPpbr-4aiN7oZeXRdmWMOl9Rqy?usp=sharing

in the data folder.

## Run testing
change the test_dir path to your path of intrest

change the path of the model to load to where you have saved it