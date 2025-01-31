# Prototype-Augmented Hypernetworks for Continual Multitask Learning

![PAH_scheme](https://github.com/user-attachments/assets/fafbe56a-a5c4-472d-8fcb-c45fd522ee78)

_Continual learning aims to learn sequential tasks
without forgetting prior knowledge, but catas-
trophic forgetting—primarily concentrated in the
final classification layers—remains a challenge.
We propose Prototype-Augmented Hypernetworks
(PAH), a framework that uses hypernetworks
conditioned on learnable task prototypes to dy-
namically generate task-specific classifier heads.
By aligning these heads with evolving represen-
tations and preserving shared knowledge through
distillation, PAH effectively mitigates catas-
trophic forgetting. Extensive evaluations on Split-
CIFAR100 and TinyImageNet demonstrate state-
of-the-art performance, achieving robust accuracy
and minimal forgetting._

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
  - [Clone the Repository](#clone-the-repository)
  - [Create a Virtual Environment](#create-a-virtual-environment)
  - [Install Dependencies](#install-dependencies)
- [Data](#data)
- [Usage](#usage)
- [Results](#results)


---

## Introduction

**Prototype-Augmented Hypernetworks (PAH)** introduce a novel approach to mitigate forgetting by dynamically generating task-specific classifier weights using a **hypernetwork** conditioned on **learnable task prototypes**. These prototypes capture task-specific characteristics, allowing the model to adapt to new tasks while maintaining performance on past tasks.

## 🔹 Key Features
- **Hypernetwork-based Classifier Adaptation**: Dynamically generates classifier heads conditioned on task-specific prototypes.
- **Prototype Learning**: Learnable task embeddings align with evolving feature representations.
- **Knowledge Distillation**: Preserves shared knowledge across tasks, reducing catastrophic forgetting.
- **State-of-the-Art Performance**: Demonstrated on **Split-CIFAR100**, **TinyImageNet**, and **Split-MNIST**, outperforming existing continual learning baselines.


## Installation

### Clone the Repository

```bash
git clone https://github.com/pah2025/PAH
cd PAH
```

### Create Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```
# Using venv
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# OR using conda
conda create -n your-env-name python=3.8
conda activate your-env-name
```

### Install Dependencies

Install the required Python packages using pip.

```
# Using pip
pip install -r requirements.txt
```

## Data

Running the training script will automatically download and save the data from Split-MNIST and Split-CIFAR100 in a new folder called data. The TinyImageNet dataset can be downloaded from the following website: https://paperswithcode.com/dataset/tiny-imagenet

## Usage

To run a training experiment the config/hyper2d.py must be modified to use the desired parameters and settings. Then run the following command to start training the model:

```
python train_hyper2d.py config/hyper2d.py
```

## Results

The obtained results will be available in a folder called results and can also be seen in Weights & Biases if your credentials are given in the wandb init function.


