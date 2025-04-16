# Comparing-GAN-Architectures
This project evaluates and compares three popular GAN loss functions—Least Squares GAN (LS-GAN), Wasserstein GAN (WGAN), and Wasserstein GAN with Gradient Penalty (WGAN-GP)—on the MedMNIST dataset using PyTorch.

## Objective
To analyze the effect of different GAN loss functions on the quality and diversity of generated medical images through:

1. Inception Score (IS)
2. Fréchet Inception Distance (FID)
3. Visual inspection

## Features
PyTorch-based GAN training loop supporting:
1. LS-GAN
2. WGAN
3. WGAN-GP

## Overview of the Code
The Jupyter notebook `ganslab-experiment4.ipynb` is organized into the following key sections:

1. Imports and Setup
  * Required packages are imported (torch, medmnist, torchvision, numpy, matplotlib, etc.).
  *TensorBoard logging and device configuration (GPU/CPU) are initialized.

2. Dataset Loading
  * The MedMNIST dataset is loaded using the medmnist API.
  * Dataloaders are created for both training and evaluation.
  * Basic normalization and transformation are applied to images.

3. Model Definitions
  * Generator and Discriminator models are defined using convolutional architectures.
  * The models are designed to be simple and flexible for experimentation.

4. Loss Functions
Three types of loss functions are implemented:
  * **LS-GAN**: Uses MSE for discriminator loss to stabilize training.
  * **WGAN**: Implements Wasserstein loss with weight clipping.
  * **WGAN-GP**: Adds gradient penalty to enforce the Lipschitz constraint instead of clipping.

5. Training Loop
  * A unified training loop is written to support all three GAN variants.
  * Loss type is selected via a configuration flag (e.g., loss_type = 'wgan-gp').
  * Each GAN variant is trained for at least 50 epochs.
  * Generator and discriminator losses are logged for analysis.

6. Evaluation Metrics
  * Inception Score (IS) and Fréchet Inception Distance (FID) are computed at regular intervals.
  * Generated samples are saved periodically for visual comparison.
  * TensorBoard logs are created for tracking performance across epochs.

7. Visualization
  * Plots of generated images and loss curves help with qualitative evaluation.
  * Visual outputs for each GAN variant can be compared side-by-side.
