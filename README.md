# Unifying heterogenous EHR datasets through a common text encoding

As part of a study project in the course CS598 Deep Learning for Healthcare at UIUC, we are replicating a publication by [Hur et al.](http://arxiv.org/abs/2108.03625) (10.48550/arXiv.2108.03625) called "Unifying Heterogeneous EHR Systems via Text-Based Code Embedding". The idea is to be able to pool electronic health records from multiple datasets using incompatible medical codes through a common text encoding. The goal is to develop a model that can take in EHR data from different sources and output a common representation of the data. This common representation can then be used for downstream tasks such as prediction of patient outcomes.

## Setup

We provide a simple setup script to install the necessary dependencies. To run the setup script, execute the following command:

```bash
bash setup.sh
```

This will install the necessary dependencies in a virtual environment. To activate the virtual environment, run the following command:

```bash
source .venv/bin/activate
```

## Data

We used the MIMIC-III dataset for our experiments. The MIMIC-III dataset is a freely accessible critical care database. To download the MIMIC-III dataset, you need to request access to the dataset from the [MIMIC-III website](https://mimic.physionet.org/gettingstarted/access/). Another dataset we used is eICU, which is a freely accessible critical care database. You can request access to the dataset from the [eICU website](https://eicu-crd.mit.edu/gettingstarted/access/). Due to the privacy concerns, we cannot provide the datasets in this repository.

## Code

The code for the project is available in the `project_code` directory. The code is organized as follows:

```bash
├── 00_preprocess # Preprocessing scripts modified from the original codebase
├── 01_pretraining # Pretraining scripts modified from the original codebase
├── 02_single_domain_learning # Single domain learning scripts modified from the original codebase
├── 03_transfer_learning # Transfer learning scripts modified from the original codebase
├── 04_pooled_learning # Pooled learning scripts modified from the original codebase
├── DescEmb # Code for the description embedding model modified from the original codebase (see license therein)
└── evaluate  # Code for experiment tracking 
```

Further details are found in [00_DL4H_Team_86_normank2_ajangid2_daniel43.ipynb](00_DL4H_Team_86_normank2_ajangid2_daniel43.ipynb).

## License

The original code by the authors was published under the MIT License. Our modifications to the code are also published under the MIT License. The original code can be found [here](https://github.com/hoon9405/DescEmb).
