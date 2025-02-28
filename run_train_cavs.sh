#!/bin/bash

# Set paths for config, latents, labels, and concepts
CONFIG_PATH="configs/config.yaml" 
LATENTS_PATH="data/activations_train.pth"
LABELS_PATH="data/labels_train.pth"
CONCEPTS_PATH="data/concept_names.pkl"
SAVE_DIR="results"

# Construct the command
eval "python -m scripts.train_cavs --config $CONFIG_PATH --latents $LATENTS_PATH --labels $LABELS_PATH --concepts $CONCEPTS_PATH --save_dir $SAVE_DIR"