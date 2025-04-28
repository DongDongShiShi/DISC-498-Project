
# Diffusion model on PDES

This project implements a generative PDE-solving approach for Darcy Flow using diffusion models, inspired by the DiffusionPDE method presented at NeurIPS 2024.

## Project Description

We address the challenge of solving partial differential equations (PDEs) under sparse observations.  
Rather than relying solely on supervised learning or physics-informed loss functions, we use a pre-trained diffusion model to sample complete solutions guided by partial observations and PDE constraints.

Specifically for Darcy Flow:
- The goal is to recover both the **coefficient field** (permeability) and the **solution field** (pressure) given very limited measurement points.
- During training, the model learns the joint distribution of Darcy coefficients and solutions from full data.
- During inference, we guide the generative process using sparse observations and enforce PDE residuals through physics-based guidance.

Our method enables high-quality reconstruction even when observations are random, grid-based, or concentrated.

## Data Source

- Raw training data is stored in `.mat` files under `data/training/darcy/`.
- Each `.mat` file contains:
  - `thresh_a_data`: Darcy coefficient fields.
  - `thresh_p_data`: Darcy solution fields.
- Each sample is a 2D field of size `(128, 128)`.
- `merge_data.py` transforms and normalizes the raw data into `.npy` files stored under `data/Darcy-merged/`.

## Required Packages

Here is the exact environment specification:

```yaml
name: Diffusion_498
channels:
  - pytorch
  - nvidia
dependencies:
  - python>=3.8, <3.10
  - pip
  - numpy>=1.20
  - click>=8.0
  - pillow>=8.3.1
  - scipy>=1.7.1
  - pytorch=1.12.1
  - psutil
  - requests
  - tqdm
  - imageio
  - pip:
    - imageio-ffmpeg>=0.4.3
    - pyspng
    - pyyaml
```

To create and activate the environment:

```bash
conda env create -f environment.yml -n Diffusion_cse498
conda activate Diffusion_cse498
```

Or manually install:

```bash
pip install numpy click pillow scipy torch psutil requests tqdm imageio imageio-ffmpeg pyspng pyyaml
```

## Instructions to Run the Code

### 1. Preprocess Data

Process raw `.mat` data into `.npy` training samples:

```bash
python merge_data.py
```

This generates `data/Darcy-merged/` containing about 50,000 `.npy` samples.

### 2. Train the Diffusion Model

Train a new diffusion model:

```bash
python train.py --outdir pretrained-darcy-new --data data/Darcy-merged/ --cond 0 --arch ddpmpp --batch 60 --batch-gpu 20 --tick 10 --snap 50 --dump 100 --duration 20 --ema 0.05
```

- `--outdir`: Where trained models are saved.
- `--data`: Preprocessed `.npy` dataset path.
- `--duration`: Number of thousands of images to train on.

### 3. Solve PDEs with Sparse Observations

Use the trained model for PDE generation:

```bash
python generate_pde.py --config config/darcy-forward.yaml
```

This will:
- Load the pretrained model.
- Apply sparse observation masks.
- Generate full coefficient and solution fields.
- Save results to output `.mat` files.

Modify `generate_pde.py` to experiment with different sampling patterns (random, grid, clustered).


---
