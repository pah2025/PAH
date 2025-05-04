# Prototype-Augmented Hypernetworks (PAH)

## Abstract

Continual learning aims to learn sequential tasks without forgetting prior knowledge, but catastrophic forgetting—primarily concentrated in the final classification layers—remains a challenge. We propose Prototype-Augmented Hypernetworks (PAH), a framework that uses hypernetworks conditioned on learnable task prototypes to dynamically generate task-specific classifier heads. By aligning these heads with evolving representations and preserving shared knowledge through distillation, PAH effectively mitigates catastrophic forgetting. Extensive evaluations on Split-CIFAR100 and TinyImageNet demonstrate state-of-the-art performance, achieving robust accuracy and minimal forgetting.

## Features

- **Dynamic Classifier Generation**: Uses hypernetworks conditioned on task prototypes to generate task-specific classifier heads.
- **Knowledge Distillation**: Preserves shared knowledge across tasks to reduce forgetting.
- **State-of-the-Art Performance**: Achieves robust accuracy and minimal forgetting on benchmarks like Split-CIFAR100 and TinyImageNet.

## How It Works

PAH employs a hypernetwork that takes task-specific prototypes as input to generate classifier heads dynamically. This approach allows the model to adapt to new tasks while retaining knowledge from previous ones. Knowledge distillation techniques are used to ensure that shared representations are preserved across tasks.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/pah2025/PAH.git
   cd PAH
  ```
2. Install the required dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Usage

...
