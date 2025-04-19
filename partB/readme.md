

## Part B - VGG16-Based Image Classification on iNaturalist_12K

This reporsitory fine-tunes a pre-trained **VGG16** model for image classification on a subset of the iNaturalist dataset.

### Setup

- **Model**: VGG16 (pretrained on ImageNet)
- **Dataset**: iNaturalist_12K (10 classes)
- **Framework**: PyTorch

---

### Model Configuration

- Loaded `vgg16` from `torchvision.models` with pretrained weights.
- **Froze convolutional layers** (`features`) to retain base representations.
- Replaced the final classification layer to match 10 output classes:
  ```python
  vgg16.classifier[6] = nn.Linear(4096, 10)
  ```
- Optimized only the classifier using **Adam** with `lr = 1e-4`.

---

### Dataset & Preprocessing

- **Transforms**:
  - Resize to `224x224`
  - Convert to tensor
  - Normalize to `[-1, 1]` range using `mean=[0.5]*3`, `std=[0.5]*3`

- **Data Sampling**:
  - **Training Set**: 5000 samples
  - **Validation Set**: 1000 samples
  - **Test Set**: 1000 samples

- **DataLoader**:
  - Batch size: `64`
  - Shuffling enabled for training set

---

### Training

- Trained for **15 epochs**.
- For each epoch:
  - Computed **training loss** and **training accuracy**
  - Evaluated on the **validation set**

---

### Evaluation

- Evaluated final model performance on the **test set** (1000 samples).


---
