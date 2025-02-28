# Concept Disentanglement
CAV Disentanglement Through Non-Orthogonality Penalization

## Description
This repository provides tools to train and analyze Concept Activation Vectors (CAVs) under orthogonality constraints designed to encourage disentanglement among concepts. It includes training scripts, metrics, and visualizations to evaluate disentanglement performance.

## Abstract
Concept Activation Vectors (CAVs) are widely used to model human-understandable concepts as directions within the latent space of neural networks. They are trained by identifying directions from the activations of concept samples to those of non-concept samples. However, this method often produces similar, non-orthogonal directions for correlated concepts, such as ''beard'' and ''necktie'' within the CelebA dataset, which frequently co-occur in images of men. This entanglement complicates the interpretation of concepts in isolation and can lead to undesired effects in CAV applications, such as activation steering.
To address this issue, we introduce a post-hoc concept disentanglement method that employs a non-orthogonality loss, facilitating the identification of orthogonal concept directions while preserving directional correctness. We evaluate our approach with real-world and controlled correlated concepts in CelebA and a synthetic FunnyBirds dataset with VGG16 and ResNet18 architectures. We further demonstrate the superiority of orthogonalized concept representations in activation steering tasks, allowing (1) the *insertion* of isolated concepts into input images through generative models and (2) the  *removal* of concepts for effective shortcut suppression with reduced impact on correlated concepts in comparison to baseline CAVs.

## Table of Contents
- [Concept Disentanglement](#concept-disentanglement)
  - [Description](#description)
  - [Abstract](#abstract)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Running](#running)

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/erenerogullari/cav-disentanglement.git
   ```

2. Navigate to the repository folder:
   ```bash
   cd cav-disentanglement
   ```

3. Install the required dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Running 
1. To train CAVs create a config.yaml file in 'configs/'
2. Run the script with the following command
  ```bash
  python -m scripts.train_cavs --config "configs/config.yaml" --latents "data/activations_train.pth" --labels "data/labels_train.pth" --concepts "data/concept_names.pkl" --save_dir "results"
  ```