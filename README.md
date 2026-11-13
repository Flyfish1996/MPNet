# MPNet: A Robust and Efficient Manifold Pooling Network for Multi-Rhythm EEG Signal Decoding

Official PyTorch implementation of **MPNet**, proposed in our paper:

**MPNet: A Robust and Efficient Manifold Pooling Network for Multi-Rhythm EEG Signal Decoding**

Guoqing Cai, Kai Zeng, Shoulin Huang, and Ting Ma

*IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2026*

[[Paper](https://doi.org/10.1109/ICASSP55912.2026.11460660)]

## Overview

MPNet is a lightweight Riemannian deep learning framework for motor imagery EEG decoding. It is designed to efficiently capture multi-rhythm temporal dynamics and spatial relationships while reducing the computational overhead of conventional deep Riemannian networks.

The key idea is to aggregate multiple time-frequency symmetric positive definite (SPD) matrices into a single, fixed-dimensional SPD representation through **Manifold Node Pooling**. This allows the subsequent Riemannian network to learn discriminative representations without separately processing every SPD node.

MPNet consists of three main components:

1. **Rhythm-Adaptive Convolutional Module:** Extracts spatial and temporal features from multiple frequency bands and models cross-frequency interactions through a coupling branch.
2. **Manifold Node Pooling Module:** Constructs SPD matrices from windowed feature representations and aggregates them into a compact SPD fusion node.
3. **Riemannian Representation Learning Module:** Employs BiMap, ReEig, and LogEig layers to learn discriminative SPD representations for classification.

## Architecture

The overall processing pipeline is:

**EEG Signals → Multi-Band Filtering → Spatial & Temporal Convolutions → Cross-Frequency Coupling → Windowed SPD Matrices → Manifold Node Pooling → SPDNet → Classification**

In the paper, EEG signals are decomposed into two frequency bands (4–17 Hz and 13–40 Hz). The resulting low-frequency, high-frequency, and coupled feature streams are each divided into four temporal windows, producing 12 SPD nodes.

Manifold Node Pooling aggregates these nodes into a single SPD matrix, which is subsequently processed by a two-block Riemannian network:

`BiMap (60 → 30) → ReEig → BiMap (30 → 15) → ReEig → LogEig → Classifier`

### Supported Pooling Strategies

MPNet supports four manifold pooling strategies:

| Method | Description |
|---|---|
| EM | Euclidean Mean Pooling |
| WEM | Weighted Euclidean Mean Pooling |
| RM | Log-Euclidean Riemannian Mean Pooling |
| WRM | Weighted Log-Euclidean Riemannian Mean Pooling |

The implementation also includes a `none` option for experiments without manifold pooling.

**Euclidean Mean Pooling (EM)** is used as the default strategy in the paper, offering a favorable balance between classification performance and computational efficiency.

## Repository Structure

```text
MPNet/
├── mpnet.py
└── spdnet_utils/
     └── ...
└── README.md
```

- `mpnet.py`: MPNet architecture, frequency-band preprocessing, manifold pooling operations, classification loss, and optimizer configuration.
- `spdnet_utils/`: Riemannian neural network components and supporting mathematical operations, including BiMap, ReEig, LogEig, and manifold-aware optimization.

## Requirements

The main dependencies are:

- Python 3
- PyTorch
- NumPy
- SciPy

Install the core dependencies using:

```bash
pip install torch numpy scipy
```

Additional dependencies may be required depending on the provided `spdnet_utils` implementation.

## Quick Start

The following example demonstrates how to initialize MPNet and perform a forward pass using synthetic EEG data.

```python
import torch
from models.mpnet import FilterBank, MPNet

# Example EEG input: (batch, channels, time)
# 4 seconds, 250 Hz, 22 EEG channels
x = torch.randn(8, 22, 1000)

# Frequency-band decomposition
filter_bank = FilterBank(
    sfreq=250,
    bands=((4, 17), (13, 40)),
    order=5,
)

x_bands = filter_bank(x)

# Initialize MPNet
model = MPNet(
    n_chans=22,
    n_classes=4,
    n_bands=2,
    hidden_channels=60,
    temporal_kernel=32,
    n_windows=4,
    pooling="EM",
)

# Forward pass
model.eval()
with torch.no_grad():
    logits = model(x_bands)

print("Output shape:", logits.shape)
# torch.Size([8, 4])
```

The filter bank transforms EEG signals from `(batch, channels, time)` into `(batch, bands, channels, time)`.

The MPNet model takes the multi-band representations as input and outputs class logits.

The example uses the architecture settings described in the paper. Other model configurations and pooling strategies can be explored through the corresponding constructor arguments.

## Experimental Evaluation

MPNet was evaluated on two publicly available motor imagery EEG datasets:

- **BCI Competition IV 2a:** 9 subjects, 4 motor imagery classes, 22 EEG channels.
- **OpenBMI:** 54 subjects, 2 motor imagery classes, 62 EEG channels.

The paper uses a cross-session evaluation protocol, with the first session for training and the second session for testing. EEG segments from 0–4 seconds after cue onset are used, and the OpenBMI signals are downsampled to 250 Hz.

### Main Results

The following results are reported in the paper using the default EM pooling strategy.

| Dataset | Accuracy (%) | Kappa (%) | F1-score (%) |
|---|---:|---:|---:|
| BCI Competition IV 2a | 80.71 ± 11.35 | 74.28 ± 15.13 | 80.37 ± 11.78 |
| OpenBMI | 79.60 ± 13.18 | 59.20 ± 17.57 | 79.33 ± 13.49 |

MPNet demonstrates competitive classification performance, reduced computational complexity, and robustness under limited training data.

For detailed comparisons, ablation studies, and efficiency analysis, please refer to the paper.

**Note:** The values above are the results reported in the publication; the provided model implementation alone does not constitute a complete reproduction pipeline.

## Implementation Notes

- MPNet supports multiple frequency bands and configurable temporal window partitioning.
- Frequency-band preprocessing uses Butterworth filtering implemented with SciPy.
- The fixed filter bank is non-differentiable and operates on the CPU.
- Both shared and independent convolutional branches are supported through the model configuration.
- Manifold Node Pooling reduces multiple SPD nodes to a single SPD representation before deep Riemannian processing.
- Riemannian representation learning and manifold-aware optimization are implemented using the accompanying `spdnet_utils` package.
- The model supports configurable pooling strategies for ablation and extension studies.

## Citation

If you find MPNet useful in your research, please cite our paper:

```bibtex
@inproceedings{cai2026mpnet,
  title={MPNet: A Robust and Efficient Manifold Pooling Network for Multi-Rhythm EEG Signal Decoding},
  author={Cai, Guoqing and Zeng, Kai and Huang, Shoulin and Ma, Ting},
  booktitle={2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={6836--6840},
  year={2026},
  publisher={IEEE},
  doi={10.1109/ICASSP55912.2026.11460660}
}
```

## Acknowledgments

This implementation builds upon research on deep Riemannian networks and SPD matrix learning. We acknowledge the contributions of prior work on SPDNet and Riemannian geometry-based EEG decoding.

## Contact

For questions or discussions regarding MPNet, please open an issue in this repository.
