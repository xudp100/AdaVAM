# AdaVAM: Adaptive Variance-Aware Momentum for Accelerating Deep Neural Network Training
## Introduction
**AdaVAM** (Adaptive Variance-Aware Momentum) is a novel optimizer designed to accelerate deep neural network training by dynamically adapting momentum based on gradient variance statistics. This repository demonstrates the effectiveness of AdaVAM for CIFAR-100 classification on ResNet-18 model.

## Project Structure
	classification_cifar100/
	├── Model/
	│ └── Resnet.py
	├── Optimizer/
	│ └── AdaVAM.py
	└── main.py

## Installation
Requires Python 3.8+ and PyTorch:

```bash
pip install torch, numpy
```
This will install the following dependencies:
* [torch](https://pytorch.org/) (the library was tested on version 1.12.1+cu113)
* [numpy](https://numpy.org/) (the library was tested on version 1.26.1)



## Running Experiments
Basic Training

```bash
python classification_cifar100/main.py --optimizer AdaVAM
```
