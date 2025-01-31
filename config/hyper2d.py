### HyperCMTL training Configuration File

import sys
import os
# Add the root of the project 
root = os.path.dirname(os.path.abspath(__file__))

# 1. Dataset Parameters
# ----------------------
dataset_config = {
    "dataset": "Split-MNIST",  # Dataset used for training. You can switch to "Split-MNIST" or other datasets.
    "NUM_TASKS": 5,  # Number of tasks for the dataset. Typically 5 for Split-MNIST and 10 for Split-CIFAR100.
    "BATCH_SIZE": 64,  # Batch size used during training.
    "VAL_FRAC": 0.1,  # Fraction of the dataset to be used for validation.
    "TEST_FRAC": 0.1,  # Fraction of the dataset to be used for testing.
    'data_dir': "./data"
}

lr_config = {
        "hyper_emb": 1e-3,  # Learning rate for the hyper-embedding network.
        "hyper_emb_reg": 0,  # Learning rate for the hyper-embedding network.
        "backbone": 1e-4,  # Learning rate for the backbone network.
        "backbone_reg": 0,  # Learning rate for the backbone network.
        "task_head": 1e-3,  # Learning rate for the task head.
        "task_head_reg": 0,  # Learning rate for the task head.
        "hypernet": 1e-3,  # Learning rate for the hypernetwork.
        "hypernet_reg": 0,  # Learning rate for the hypernetwork.
}

# 2. Model Hyperparameters
# ------------------------
model_config = {
    "backbone": "mobilenetv2",  # Backbone architecture used for the model (e.g., "resnet50", "mobilenet"). 
    "hyper_hidden_features": 16,
    "hyper_hidden_layers": 1,
    "frozen_backbone": False,  # Whether to freeze the backbone during training.
    "prototypes_channels": 1, # Number of channels of prototypes 1 for grayscale, 3 for RGB
    "prototypes_size": 10,  # Size of the prototypes.
    "initialize_prot_w_images": True,
    "mean_initialization_prototypes": 0.5,  # Mean for the initialization of the prototypes.
    "std_initialization_prototypes": 0.1,  # Standard deviation for the initialization of the prototypes.
    "lr_config": lr_config}


# 3. Training Parameters
# -----------------------
training_config = {
    "epochs_per_timestep": 15,  # Number of epochs per timestep (task).
    "temperature": 2.0,  # Temperature for distillation loss (used in knowledge distillation).
    "stability": 0.5,  # Stability weight for soft distillation loss.
    "weight_hard_loss_prototypes": 0,  # Weight for the hard loss applied to the prototypes.
    "weight_soft_loss_prototypes": 1,  # Weight for the soft loss applied to the prototypes.
    "weight_smoothness_loss": 0,
    "optimizer": "AdamW",  # Optimizer used for training. AdamW is used here.
}


# 5. Logging and Visualization Parameters
# ---------------------------------------
frozen = "frozen" if model_config["frozen_backbone"] else ""
name = f"Hyper2d-{frozen}-{model_config['backbone']}-{dataset_config['dataset']}--initialized={model_config['initialize_prot_w_images']}"
logging_config = {
    "log_file": "training.log",  # Log file where training information will be saved.
    "log_level": "INFO",  # Logging level for the training process (can be INFO, DEBUG, etc.).
    "plot_training": True,  # Whether to plot the training curves.
    "show_progress": True,  # Whether to show progress bars during training.
    "verbose": True,  # Whether to show detailed logs for each epoch.
    "results_dir": "results",  # Folder to save the results.
    "name": name,  # or EWC_Baseline pr SI_Baseline,
    "group": "Hyper2d"
}

# 6. Miscellaneous Parameters
# ---------------------------
misc_config = {
    "device": "cuda",  # Device for training (use "cpu" if no GPU is available).
    "seed": 1,  # Seed for reproducibility.
}

# 7. Evaluation Parameters
# ------------------------
evaluation_config = {
    "eval_frequency": 1,  # Frequency of evaluation (e.g., every epoch).
    "plot_results": True,  # Whether to plot results after each timestep.
}



# Combine all config sections into one dictionary for easy access
config = {
    "dataset": dataset_config,
    "model": model_config,
    "training": training_config,
    "logging": logging_config,
    "misc": misc_config,
    "evaluation": evaluation_config,
}
