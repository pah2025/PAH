# Prototype-Augmented Hypernetworks for Scalable Continual Multitask Learning

![PAH_scheme](https://github.com/user-attachments/assets/fafbe56a-a5c4-472d-8fcb-c45fd522ee78)

_Continual learning aims to learn sequential tasks
without forgetting prior knowledge, but catas-
trophic forgetting—primarily concentrated in the
final classification layers—remains a challenge.
We propose Prototype-Augmented Hypernetworks
(P AH), a framework that uses hypernetworks
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

- [Project Title](#project-title)
- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Clone the Repository](#clone-the-repository)
  - [Create a Virtual Environment](#create-a-virtual-environment)
  - [Install Dependencies](#install-dependencies)
- [Data](#data)
  - [Downloading the Data](#downloading-the-data)
  - [Data Preprocessing](#data-preprocessing)
- [Usage](#usage)
  - [Running Experiments](#running-experiments)
  - [Training the Model](#training-the-model)
  - [Evaluating the Model](#evaluating-the-model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Introduction

Provide a more detailed introduction to your project. Explain the problem it solves, the motivation behind it, and any relevant background information.

## Features

- **Feature 1:** Description of feature 1.
- **Feature 2:** Description of feature 2.
- **Feature 3:** Description of feature 3.
- *(Add more features as needed)*

## Installation

### Prerequisites

List the software and tools required to run your project. For example:

- **Python 3.8+**
- **Git**
- **CUDA 11.0** _(if using GPU)_
- *(Add any other prerequisites)*

### Clone the Repository

```bash
git clone https://github.com/pah2025/PAH
cd PAH
```

### Create Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```
# Using venv
python3 -m venv env
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





