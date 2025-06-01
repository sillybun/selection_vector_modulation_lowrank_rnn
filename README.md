# Selection Vector Modulation Low-Rank RNN

This repository contains the official implementation of the paper "Elucidating the Selection Mechanisms in Context-Dependent Computation through Low-Rank Neural Network Modeling" (eLife, 2024). The project implements a low-rank Recurrent Neural Network (RNN) model with selection vector modulation to study neural network dynamics and information processing using low-rank approximations.

## Paper Abstract

This repository provides the code implementation for our research on selection mechanisms in context-dependent computation. The paper has been assessed as important and convincing by eLife.

## Project Structure

```
.
├── model.py           # Core model implementations (lowrank_encoder, reparameter_encoder)
├── dataset.py         # Dataset and input generation utilities
├── toolfunc.py        # Helper functions and utilities
├── requirements.txt   # Project dependencies
├── figure/           # Directory containing generated figures
└── *.ipynb           # Jupyter notebooks for experiments and visualizations
```

## Requirements

The project requires the following main dependencies:

```
torch>=2.0.0
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
tqdm>=4.62.0
jupyter>=1.0.0
ipython>=7.0.0
torchfunction
zytlib
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/selection_vector_modulation_lowrank_rnn.git
cd selection_vector_modulation_lowrank_rnn
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

Note: `torchfunction` and `zytlib` are custom packages that may need to be installed separately.

## Usage

The project includes several Jupyter notebooks demonstrating different experiments and analyses:

- `Figure 2.ipynb`: Experiments and visualizations for Figure 2
- `Figure 3.ipynb`: Experiments and visualizations for Figure 3
- `Figure 5.ipynb`: Experiments and visualizations for Figure 5

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{zhang2024elucidating,
  title={Elucidating the Selection Mechanisms in Context-Dependent Computation through Low-Rank Neural Network Modeling},
  author={Zhang, Y and Feng, J and Min, B},
  journal={eLife},
  year={2024},
  doi={10.7554/eLife.103636.1}
}
```