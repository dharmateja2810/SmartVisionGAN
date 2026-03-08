# SRGAN - Super-Resolution Generative Adversarial Network

This project implements SRGAN (Ledig et al., 2017) to upscale low-resolution CT scan images by 4x using a combination of adversarial loss and VGG-based perceptual loss. The model is trained on the SARS-CoV-2 CT Scan Dataset from Kaggle.



## Project Structure

```
SRGAN/
  model.py                  Model definitions and dataset class
  data_exploration.ipynb    Dataset exploration and visualization
  training.ipynb            Model training with loss tracking
  evaluation.ipynb          Evaluation, metrics, and visual comparison
  checkpoints/              Saved model weights during and after training
  README.md                 This file
```

## Dataset

**Source:** SARS-CoV-2 CT Scan Dataset (Kaggle, by Plamen Eduardo)

The dataset contains 2,481 CT scan images split into two classes: COVID (1,252 images) and non-COVID (1,229 images). All images are in PNG format with an average file size of 95.3 KB. Original resolutions range from 182x129 to 488x408 pixels, with a median size of 351x259 pixels.

For training, each image is resized to produce a paired sample:

- **High-Resolution (HR):** 256x256 pixels (ground truth target)
- **Low-Resolution (LR):** 64x64 pixels (bicubic downscale, 4x reduction)

The generator learns the mapping from 64x64 input to 256x256 output.

## Model Architecture

### Generator (SRResNet-based)

The generator follows the SRResNet architecture from the original SRGAN paper. It begins with an initial 9x9 convolution with 64 filters and PReLU activation. This feeds into 16 residual blocks, where each block contains two 3x3 convolutions with batch normalization and PReLU, along with a skip connection. After the residual blocks, a 3x3 convolution with batch normalization is applied and added back to the initial features via a global skip connection. Upsampling is performed by two sub-pixel convolution layers (PixelShuffle x2 each, giving 4x total upscaling). The final output layer is a 9x9 convolution with 3 output channels and sigmoid activation.

Input shape: (B, 3, 64, 64). Output shape: (B, 3, 256, 256). Values in [0, 1]. Total trainable parameters: approximately 1.5M.

### Discriminator (VGG-style)

A binary classifier that distinguishes real HR images from generator outputs. It starts with a 3x3 convolution with 64 filters and LeakyReLU(0.2) activation. The feature extraction stage consists of 8 convolutional blocks with increasing channel counts (64, 128, 256, 512) using alternating stride-1 and stride-2 layers. Each block uses Conv2d followed by BatchNorm and LeakyReLU(0.2). After feature extraction, adaptive average pooling reduces the spatial dimensions to 1x1. The classification head uses two 1x1 convolutions (512 to 1024 to 1) with a final sigmoid activation.

Input shape: (B, 3, 256, 256). Output shape: (B,) with values in [0, 1].

### VGG Feature Extractor

A frozen VGG19 network (pretrained on ImageNet) truncated at layer 36 (conv5_4). Used solely for computing perceptual loss. Input images are normalized with ImageNet statistics before feature extraction.

## Loss Functions

### Generator Loss

The generator is trained with two loss components:

1. **Perceptual loss (VGG feature loss):** MSE between VGG19 features of the generated image and the ground truth HR image. This encourages the generator to produce outputs that are perceptually similar to real images rather than just pixel-accurate.

2. **Adversarial loss:** Binary cross-entropy loss encouraging the discriminator to classify generated images as real.

Total generator loss = perceptual_loss + 0.001 * adversarial_loss

### Discriminator Loss

Standard binary cross-entropy averaged over real and fake predictions:

discriminator_loss = 0.5 * (BCE(D(real), 1) + BCE(D(fake), 0))

## Training Configuration

The model is trained for 20 epochs with a batch size of 8 using the Adam optimizer with a learning rate of 1e-4. The adversarial loss weight is set to 1e-3. Training runs on CUDA (GPU) if available, otherwise it falls back to CPU.

Checkpoints are saved every 5 epochs and at the end of training. Each checkpoint contains generator weights, discriminator weights, optimizer states, and the full training loss history.

## Pipeline

The project is organized as a three-step pipeline:

### Step 1: Data Exploration (data_exploration.ipynb)

- Dataset statistics (image count, format, file sizes)
- Class distribution (COVID vs non-COVID)
- Original resolution analysis with scatter plots
- 4x4 raw image gallery
- Side-by-side LR (64x64) and HR (256x256) paired samples
- RGB channel intensity histograms for LR and HR
- Bicubic upscale baseline vs ground truth with PSNR values
- Per-pixel mean image and variance heatmap across the dataset

### Step 2: Training (training.ipynb)

- Loads the dataset and model classes from model.py
- Architecture verification with forward pass sanity check
- Full training loop with per-epoch loss tracking (6 metrics)
- Checkpoint saving every 5 epochs
- Training loss curves (G vs D, perceptual vs adversarial, D real vs D fake)
- Post-training inference sanity check

### Step 3: Evaluation (evaluation.ipynb)

- Loads trained model from checkpoint
- Training loss curve visualization from saved history
- Qualitative comparison grid: LR vs Bicubic vs SRGAN vs HR
- Full-dataset PSNR and SSIM computation for both Bicubic and SRGAN
- PSNR and SSIM distribution histograms
- Zoomed-in centre crop comparison for texture detail
- Pixel-wise error heatmaps (SRGAN error vs Bicubic error)
- Single-image inference demo with a reusable function
- FFT frequency domain analysis comparing spectral content

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- Pillow
- matplotlib
- numpy
- tqdm

Install all dependencies:

```
pip install torch torchvision pillow tqdm matplotlib numpy
```

## Usage

1. Download the SARS-CoV-2 CT Scan Dataset from Kaggle and extract it locally.

2. Update `DATASET_PATH` in each notebook to point to your local dataset folder.

3. Run the notebooks in order:
   - `data_exploration.ipynb` to inspect the data
   - `training.ipynb` to train the model (saves weights to `checkpoints/`)
   - `evaluation.ipynb` to evaluate and visualize results (loads from `checkpoints/`)

## Results

The evaluation notebook computes PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index) across the full dataset, comparing SRGAN output against both the bicubic upscale baseline and the ground truth HR images.

Key observations from data exploration:
- CT scans are effectively grayscale (R, G, B channels are identical)
- Pixel values are well distributed in [0, 1] after ToTensor normalization
- Bicubic upscaling produces visibly blurry results, providing clear room for improvement

## References

1. Ledig, C., et al. "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network." CVPR 2017.
2. Soares, E., et al. "SARS-CoV-2 CT-scan dataset: A large dataset of real patients CT scans for SARS-CoV-2 identification." medRxiv 2020.
3. Simonyan, K. and Zisserman, A. "Very Deep Convolutional Networks for Large-Scale Image Recognition." ICLR 2015.
