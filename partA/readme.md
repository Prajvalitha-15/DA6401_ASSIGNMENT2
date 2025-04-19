##  Part-A : Image Classification using Custom CNN Architectures & W&B Hyperparameter Sweeps

This repository contains two python files :

1. `q1_cnn.py`: Defines a custom lightweight CNN model architecture for image classification.
2. `q24.py`: Implements an end-to-end pipeline with dataset preprocessing, CNN training, hyperparameter tuning using [Weights & Biases (W&B)](https://wandb.ai/), and model evaluation on the iNaturalist mini dataset.

---

## `q1_cnn.py`: Small Custom CNN

This script contains a simple but modular CNN implementation called `SmallCNN`

### Key Features

- **Modular convolutional blocks** with configurable number of filters and kernel sizes.
- **Flexible activation function** (default is ReLU).
- **Dynamic computation** of the number of features going into the fully connected layer.
- **Two-layer classifier head** (dense + output layer for `num_classes` classes).

### Architecture

```
Input (3 x 224 x 224)
→ [Conv → Activation → MaxPool] x 5
→ Flatten
→ Dense Layer
→ Activation
→ Output Layer (num_classes)
```

---

##  `q24.py`: Training & Hyperparameter Tuning

This script takes the CNN training pipeline to the next level with support for:

### Hyperparameter Sweep via W&B
It defines a **random search sweep** over the following parameters:
- `filters`: Initial number of convolutional filters
- `activationFunction`: ReLU, GELU, or SiLU
- `filterOrganisation`: Same or Halved filter sizes across layers
- `batchNormalisation`: With or without BatchNorm
- `dropout`: Dropout rates

### Custom CNN Class (`CNN`)
A configurable CNN that supports:
- Adjustable filter layouts and activation functions
- Optional batch normalization and dropout
- Classifier with fully connected layers

### Dataset
- **Dataset**: iNaturalist 12K
- **Preprocessing**:
  - Resize to 224x224
  - Normalization to [-1, 1]
- **Subset used**: 
  - 6000 images for training/validation
  - 1000 for testing
- **DataLoader** is set with shuffling, batching, and device detection.

### Training
- Loss: `CrossEntropyLoss`
- Optimizer: `Adam`
- Logging: All training/validation metrics are logged with W&B per epoch.

### Sweep & Evaluation
- Runs 48 trials using `wandb.agent` in **kaggle**
- Finds the best run based on **validation accuracy**
- Trains a final model using best hyperparameters
- Evaluates on the **test set**
- Visualizes predictions for 30 test images

---

## Sample Output Visualizations

After training, the script plots 30 sample predictions along with their predicted and ground truth labels.

---

## Requirements

Install the following:
```bash
pip install torch torchvision wandb pillow kaggle
```

Instructions to run **W&B** in **kaggle**:
- Upload your `wandb-key` to Kaggle secrets as `wandb-key`
---