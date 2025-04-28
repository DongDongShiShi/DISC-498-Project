# DISC-498-Project: Data Instructions

This project uses training and testing data from the *Diffusion* paper and *PDEBench*.

## Data Sources

- **Darcy Flow, Poisson Equation, and Burgers' Equation**
  - **Training Data:** [Download Link](https://drive.google.com/file/d/1z4ypsU3JdkAsoY9Px-JSw9RS2f5StNv5/view?usp=sharing)
  - **Testing Data:** [Download Link](https://drive.google.com/file/d/1HdkeCKMLvDN_keIBTijOFYrRcA3Quy0l/view?usp=sharing)

- **Other PDEs (ReacDiff, Advection) (from PDEBench)**
  - **All Data:** [Download Link](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986)

## Data Format

All datasets are stored in the format `[M, X, Y]`, where:
- `M` = number of samples,
- `X`, `Y` = resolution dimensions for each sample.

## Setup Instructions

1. Download the datasets from the links above.
2. Unzip all downloaded files.
3. Place all extracted folders inside the `data/` directory of this project.

The folder structure should look like:

```
data/
├── training/
├── testing/
├── pdebench_data/
│   ├── advection/
│   ├── reacdiff/
│   └── ...
```

4. Run the `merge_data.py` script to convert all datasets into `.npy` files.

After running `merge_data.py`, the folder structure will be:

```
data/
├── darcy_merge/
│   ├── darcy_train.npy
│   ├── darcy_test.npy
├── poisson_merge/
│   ├── poisson_train.npy
│   ├── poisson_test.npy
├── burgers_merge/
│   ├── burgers_train.npy
│   ├── burgers_test.npy
├── pdebench_data/
│   ├── advection/
│   ├── reacdiff/
│   └── ...
```

## Notes

- Ensure that all datasets are placed correctly before running training or evaluation scripts.
- `merge_data.py` must be executed once after downloading to generate the `.npy` files required by the training pipeline.
