# DISC-498-Project

# Diffusion for PDE 

Jiuzhou Chen， Dongwei Shi
![DiffusionPDE](doc/PDEdatavisu.jpg)
(figure from [PDEBench Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/0a9747136d411fb83f0cf81820d44afb-Paper-Datasets_and_Benchmarks.pdf) given by Takamoto et.al.)

## Data Generation

All training and test datasets can be downloaded from [PDEBench Github](https://github.com/pdebench/PDEBench) and link of [dataset]( https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986)

A formal table to summarize PDE types and grids from PDE Bench Paper：

![DiffusionPDE](doc/PDEdataTable.jpg)

## Train Diffusion Models

All pre-trained models can be downloaded from [here](https://drive.google.com/file/d/1w4V0o-nTjpHP_Xv32Rt_SgPGmVa9PwL_/view?usp=sharing). Unzip the ``pretrained-models.zip`` in the root directory.

Our training script is derived from [EDM](https://github.com/NVlabs/edm). To train a new diffusion model on the joint distribution, use, e.g.,

```python
# Prepare the .npy files for training. 
# Raw data in the datasets should be scaled to (-1, 1).
python3 merge_data.py # Darcy Flow

# Train the diffusion model.
torchrun --standalone --nproc_per_node=3 train.py --outdir=pretrained-darcy-new --data=/data/Darcy-merged/ --cond=0 --arch=ddpmpp --batch=60 --batch-gpu=20 --tick=10 --snap=50 --dump=100 --duration=20 --ema=0.05
```

## Solve Forward Problem

To solve the forward problem with sparse observation on the coefficient (or initial state) space, use, e.g.,

```python
python3 generate_pde.py --config configs/darcy-forward.yaml
```

### Solve Inverse Problem

To solve the inverse problem with sparse observation on the solution (or final state) space, use, e.g.,

```python
python3 generate_pde.py --config configs/darcy-inverse.yaml
```

## Recover Both Spaces With Observation On Both Sides

To simultaneously solve coefficient (initial state) space and solution (final state) space with sparse observations on both sides, use, e.g.,

```python
python3 generate_pde.py --config configs/darcy.yaml
```

## Solve Solution Over Time

To recover the solution throughout a time interval with sparse sensors, use, e.g.,

```python
python3 generate_pde.py --config configs/burgers.yaml
```

## License

<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><span property="dct:title">DiffusionPDE: Generative PDE-Solving Under Partial Observation</span> by <span property="cc:attributionName">Jiahe Huang, Guandao Yang, Zichen Wang, Jeong Joon Park</span> is licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International</a>.</p>

``dnnlib, torch_utils, training`` folders, and ``train.py`` are derived from the [codes](https://github.com/NVlabs/edm) by Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. The codes were originally shared under the [Attribution-NonCommercial-ShareAlike 4.0 International License](https://github.com/NVlabs/edm/blob/main/LICENSE.txt).

Data generation codes for Darcy Flow, Burgers' equation, and non-bounded Navier-Stokes equation are derived from the [codes](https://neuraloperator.github.io/neuraloperator/dev/index.html) by  Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. The codes were originally shared under the [MIT license](https://github.com/neuraloperator/neuraloperator/blob/main/LICENSE).


