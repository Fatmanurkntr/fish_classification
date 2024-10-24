# fish_classification
A large scale fish classification project using Artificial Neural Networks (ANN).

## Overview
This project aims to classify various species of fish using a deep learning model built with TensorFlow and Keras. The dataset consists of **9,000 images** of different fish species, serving as the foundation for training, validating, and testing the model's performance. The ultimate goal is to accurately identify the species based on their visual characteristics, enabling potential applications in marine biology, ecology, and fisheries management.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)

## Installation
To set up this project, ensure you have Python installed on your machine along with the required libraries. You can install the necessary packages using pip:

pip install tensorflow matplotlib scikit-learn
## Data Preparation
Before training the model, the dataset undergoes several preprocessing steps:

1. **Image Resizing:** All images are resized to a uniform dimension of **128x128 pixels** to maintain consistency across the dataset. This step helps in reducing computational load and ensures that the input to the model is of the same size.

2. **Normalization:** Pixel values are normalized to a range of 0 to 1 by dividing by 255. This process helps improve the convergence speed of the model during training.

3. **Label Encoding:** The class labels are converted into a numerical format using `LabelEncoder` from the `scikit-learn` library. This encoding is necessary for the model to interpret the categorical data correctly.

4. **Data Splitting:** The dataset is divided into three subsets:
   - **Training Set:** Used to train the model (typically 70% of the dataset).
   - **Validation Set:** Used to validate the model during training and fine-tune hyperparameters (typically 15% of the dataset).
   - **Test Set:** Used to evaluate the model's final performance after training (typically 15% of the dataset).

## Model Architecture
The deep learning model follows a sequential architecture designed to process images effectively. Below is a detailed explanation of the model's structure:

### Input Layer:
- **Flatten Layer:** The images with dimensions (128, 128, 3) are flattened into a 1D vector to prepare them for the dense layers.

### Hidden Layers:
1. **First Hidden Layer:**
   - **Dense Layer (512 neurons):** Captures complex patterns in the data.
   - **Batch Normalization:** Stabilizes learning and improves convergence speed.
   - **Activation (ReLU):** Introduces non-linearity to the model.
   - **Dropout (0.3):** Reduces overfitting by randomly disabling 30% of the neurons during training.

2. **Second Hidden Layer:**
   - **Dense Layer (256 neurons):** Continues feature extraction.
   - **Batch Normalization:** Enhances training stability.
   - **Activation (ReLU):** Facilitates learning of complex features.
   - **Dropout (0.2):** Further mitigates overfitting.

3. **Third Hidden Layer:**
   - **Dense Layer (128 neurons):** Extracts more abstract features.
   - **Batch Normalization:** Applies normalization to improve robustness.
   - **Activation (ReLU):** Maintains non-linearity.
   - **Dropout (0.2):** Helps prevent overfitting.

4. **Fourth Hidden Layer:**
   - **Dense Layer (64 neurons):** Refines the learned features.
   - **Batch Normalization:** Normalizes outputs.
   - **Activation (ReLU):** Introduces non-linearity.
   - **Dropout (0.1):** Retains more information with a lower dropout rate.

5. **Fifth Hidden Layer:**
   - **Dense Layer (64 neurons):** Processes features for final classification.

### Output Layer:
- **Dense Layer (n classes, softmax):** Outputs a probability distribution across the target classes, allowing the model to predict the likelihood of each class.

## Training
The model is compiled using:
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy (suitable for multi-class classification)
- **Metrics:** Accuracy

During the training phase, the model learns to adjust its weights based on the computed loss from predictions compared to actual labels. The training process involves iterating through the training dataset multiple times (epochs) while validating against the validation set to monitor performance and prevent overfitting.

## Evaluation
After training, the model's performance is evaluated using various metrics:
- **Loss and Accuracy Graphs:** Visual representations of training and validation performance over epochs.
- **Confusion Matrix:** Displays the performance of the classification model on a set of test data.
- **Classification Report:** Provides detailed metrics such as precision, recall, and F1 score for each class.

## Results
The model achieves a significant level of accuracy of **97%** in classifying fish species, demonstrating the effectiveness of using a deep learning approach with the chosen architecture. The combination of dropout rates and batch normalization contributes to better performance and reduced overfitting.


## Contributing
Contributions are welcome! If you would like to improve the project, please fork the repository and submit a pull request.

[Fish Classification with ANN on Kaggle](https://www.kaggle.com/code/fatmanurkantar/fish-classification-with-ann)


