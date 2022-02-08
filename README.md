# Centrilsyze

A small python package for analysing images of centrioles.

## Installation

It is reccomended that you install PanDDA 2 in it's own python 3.9 anaconda enviroment. This can be achieved by installing anaconda and then:

```bash
conda create -n pandda2 python=3.9
conda activate pandda2
conda install -c conda-forge -y fire numpy scipy joblib scikit-learn umap-learn bokeh dask dask-jobqueue hdbscan matplotlib
conda install -c conda-forge -y seaborn
git clone https://github.com/ConorFWild/centrilyze.git
pip install -e .

```



## Running

Once you have installed PanDDA 2 in a conda enviroment, running it is as simple as:

```bash
python /path/to/analyse.py <data directories> <output directory> --pdb_regex="dimple.pdb" --mtz_regex="dimple.mtz" --structure_factors='("2FOFCWT","PH2FOFCWT")' <options>

```
