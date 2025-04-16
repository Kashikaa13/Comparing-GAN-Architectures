# 📊 Comparing GAN Architectures

This project evaluates and compares three popular GAN loss functions—**Least Squares GAN (LS-GAN)**, **Wasserstein GAN (WGAN)**, and **Wasserstein GAN with Gradient Penalty (WGAN-GP)**—on the **MedMNIST** dataset using PyTorch.

---

## 🎯 Objective

To analyze how different GAN loss functions affect the **quality** and **diversity** of generated medical images through:

- **Inception Score (IS)**
- **Fréchet Inception Distance (FID)**
- **Visual inspection**

---

## ⚙️ Features

- Unified **PyTorch-based** training loop.
- Support for three GAN variants:
  - LS-GAN
  - WGAN
  - WGAN-GP
- Integrated evaluation using IS and FID.
- Visualization of generated samples and training losses.
- TensorBoard logging for real-time monitoring.

---

## 📁 Code Overview: `ganslab-experiment4.ipynb`

### 1. **Imports and Setup**
- All required libraries (`torch`, `medmnist`, `torchvision`, `matplotlib`, etc.) are imported.
- GPU/CPU configuration is initialized.
- TensorBoard setup for monitoring training metrics.

### 2. **Dataset Loading**
- **MedMNIST** dataset is loaded via the `medmnist` API.
- Dataloaders are created for both training and evaluation.
- Basic image normalization and transformations applied.

### 3. **Model Definitions**
- **Generator** and **Discriminator** are defined using convolutional layers.
- Architectures are lightweight and modular for experimentation across different loss functions.

### 4. **Loss Functions**
Three loss formulations are supported:

- **LS-GAN**: Uses **Mean Squared Error (MSE)** for discriminator loss to stabilize training.
- **WGAN**: Implements **Wasserstein loss** with **weight clipping**.
- **WGAN-GP**: Adds a **gradient penalty** to enforce the Lipschitz constraint, instead of clipping.

### 5. **Training Loop**
- A single unified training loop handles all three GAN types.
- `loss_type` flag controls which GAN variant is used (e.g., `'wgan-gp'`).
- Both Generator and Discriminator are trained for at least **50 epochs**.
- All relevant metrics and losses are logged.

### 6. **Evaluation Metrics**
- **Inception Score (IS)** and **Fréchet Inception Distance (FID)** are calculated at intervals.
- Generated images are saved periodically for side-by-side comparison.
- TensorBoard logs track performance and trends over time.

### 7. **Visualization**
- Generated images from each GAN variant are plotted.
- Loss curves for Generator and Discriminator are visualized.
- Results are compared qualitatively to assess sharpness, realism, and diversity.

---




