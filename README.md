# SACSANN
Sequence-based predictor of chromosomal compartments

## Overview
The 'Sequence-based Annotator of Chromosomal Compartments by Stacked Artificial Neural Networks' or SACSANN is a machine learning approach to predicting A/B compartment annotations using only features derived from a reference genome. SACSANN has been test on both Linux and MacOS environments.

## Software requirements

1) Python 3 (v3.7.3 tested) - installed via pyenv (see below)
2) Homebrew (v2.5.1 tested) - https://brew.sh/
3) pyenv (v1.2.20 tested) - https://github.com/pyenv/pyenv (Homebrew installation recommended)
4) pyenv-virtualenv (v1.1.15 tested) - https://github.com/pyenv/pyenv-virtualenv (Homebrew installation recommended)

## Installation

To install pyenv-virtualenv and the tested version of Python for SACSANN, follow these steps:
1)  Install pyenv and pyenv-virtualenv using Homebrew
```bash
brew update
brew install pyenv
brew install pyenv-virtualenv
```

2) Install tested version of Python 3 (v3.7.3) 
```bash
pyenv install -v 3.7.3
```

To install SACSANN, follow these steps:

1) Clone the SACSANN repo
```bash
git clone https://github.com/BlanchetteLab/SACSANN
```

2) Redirect terminal to installation folder: `cd SACSANN/`

3)  Create a Python 3 virtual environment using pyenv and pyenv-virtualenv:
```bash
pyenv virtualenv 3.7.3 sacsann
pyenv local sacsann
```
4) Install additional software requirements (i.e., Python modules): 
```bash 
make install
```

## Run SACSANN demo

You can run a demo of SACSANN on two mouse chromosomes (chr1 and chr2) by running:
 ```bash
 make run-demo
 ```

Example input/target files can be found in: `data/`

For more information of available SACSANN options:
```
python sacssann.py --help
```

![Sacsann arguments](doc/sacsann_arguments.png)

## Input

* Expected input features should be stored in a CSV file, with one file per chromosome to analyze. CSV format (no header):
	* rows are genomic bins (e.g., 100 kb)
	* columns are input features (e.g., GC content, count of TFBSs or TEs)
	* See `data/features/` for examples

* For training, binary labels (i.e., A/B compartments) should also be in CSV files  (no header) with one row per genomic bin and one file per chromosome. Binary labels are interpreted by SACSANN as follows :
  - 0: bin included in a B compartment
  - 1: bin included in an A compartment
  - See `data/labels` for examples
 
## Output

* Predicted compartments for the input tests chromosomes can be found in the specified output folder (default is `output/`), with one file per chromosome
 
* If the `save_model` argument is set to `True` (`sacsann.py` optional argument), learned/used models weights and parameters will be saved in the pickle format in the following files:
  - `mlp_int_weights.p`
  - `scaler.p`
  - `mlp_top_weights.p`
  - `final_scaler.p`

## License
SACSANN is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation.

SACSANN is distributed in the hopes that it will be useful, but WITHOUT ANY WARRANTY;  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.
