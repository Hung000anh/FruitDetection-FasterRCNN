```markdown
# FruitDetection-FasterRCNN

## Faster R-CNN Object Detection on Fruits Dataset
This repository contains a Jupyter Notebook (`notebook.ipynb`) that demonstrates the implementation of a Faster R-CNN (Region-based Convolutional Neural Network) model for object detection on a fruits dataset. The dataset consists of images of fruits (oranges, apples, and bananas) with corresponding XML annotations for bounding boxes.

## Table of Contents
- [Introduction](#introduction)
- [Setup](#setup)
- [Dataset Preparation](#dataset-preparation)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Dependencies](#dependencies)

## Introduction
Faster R-CNN is a popular deep learning model used for object detection tasks. This project focuses on detecting fruits (oranges, apples, and bananas) in images using a Faster R-CNN model. The model is trained on a custom dataset of fruit images, with annotations provided in XML format.

## Setup

### 1. Mount Google Drive
The notebook assumes that the dataset is stored in Google Drive. The first step is to mount Google Drive to access the dataset.

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Install Required Libraries
The notebook uses the `torch_snippets` library for data processing and visualization. Install it using pip:

```python
!pip install -q torch_snippets
from torch_snippets import *
```

### 3. Import Necessary Libraries
The following libraries are used in the notebook:

```python
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from xml.etree import ElementTree as et
from torchvision.ops import nms
import matplotlib.pyplot as plt
```

## Dataset Preparation

### 1. Define Dataset Paths and Labels
The dataset is stored in the `train` and `test` directories within Google Drive. The labels for the dataset are defined as background, orange, apple, and banana.

```python
root = '/content/drive/MyDrive/IUH/Computer_Vision/Fruits_RCNN_Object_Detection/data/train_zip/train/'
labels = ['background', 'orange', 'apple', 'banana']
```

### 2. Create Custom Dataset Class
A custom `FruitsDataset` class is created to handle the loading and preprocessing of the dataset. The class reads images and corresponding XML annotations, resizes the images, and extracts bounding box information.

```python
class FruitsDataset(Dataset):
    def __init__(self, root=root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.img_paths = sorted(Glob(self.root + '/*.jpg'))
        self.xlm_paths = sorted(Glob(self.root + '/*.xml'))
```

### 3. Prepare DataLoader
The dataset is split into training and validation sets, and `DataLoader` objects are created for both sets.

```python
tr_ds = FruitsDataset()
tr_dl = DataLoader(tr_ds, batch_size=4, shuffle=True, collate_fn=tr_ds.collate_fn)

val_ds = FruitsDataset(root=val_root)
val_dl = DataLoader(val_ds, batch_size=2, shuffle=True, collate_fn=val_ds.collate_fn)
```

## Model Training

### 1. Define the Faster R-CNN Model
The Faster R-CNN model is loaded with a ResNet-50 backbone and a Feature Pyramid Network (FPN). The model's final layer is replaced to match the number of classes in the dataset.

```python
def get_model():
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
```

### 2. Train the Model
The model is trained using the training `DataLoader`. The training process involves forward and backward passes, and the model's performance is monitored using a Report object.

```python
model = get_model().to(device)
model(imgs, targets)
```

## Evaluation

### 1. Evaluate the Model
The model's performance is evaluated on the validation set. The evaluation metrics include loss values for classification, bounding box regression, objectness, and RPN box regression.

```python
EPOCH: 0.033  val_loss: 0.117  val_loc_loss: 0.037  val_regr_loss: 0.075  val_loss_objectness: 0.003  val_loss_rpn_box_reg: 0.001
```

## Usage
To use this notebook, follow these steps:

1. **Mount Google Drive:** Ensure that your dataset is stored in Google Drive and mount it in the notebook.
2. **Install Dependencies:** Install the required libraries using pip.
3. **Prepare Dataset:** Define the dataset paths and labels, and create the custom dataset class.
4. **Train the Model:** Define and train the Faster R-CNN model using the training `DataLoader`.
5. **Evaluate the Model:** Evaluate the model's performance on the validation set.

## Dependencies
The following dependencies are required to run the notebook:

- Python 3.x
- PyTorch
- torchvision
- torch_snippets
- PIL (Pillow)
- matplotlib

