
# Hate Speech Classification Model

This repository provides a machine learning pipeline for classifying text into three categories: `hate speech`, `offensive speech`, and `normal speech`. The model is built using TensorFlow and trained on the **hate-offensive-speech** dataset.

---

## Table of Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Project Structure](#project-structure)
5. [Dataset](#dataset)
6. [Model Details](#model-details)
7. [License](#license)

---

## Features
- Train a bidirectional LSTM-based text classifier.
- Use preprocessed datasets for training, validation, and testing.
- Tokenize input text and classify it in real-time.

---

## Usage

### 1. Train the Model
To train the model, execute the `try.py` script. It handles dataset preparation, model creation, and training:
```bash
python try.py
```

The script:
- Loads the dataset using the `datasets` library.
- Trains a bidirectional LSTM model.
- Saves the trained model to a `.h5` file (default: `hate_class.h5`).

---

### 2. Predict Text Classifications
After training, the script allows you to classify text inputs:
```bash
python try.py
```

Example:
```bash
Type text to classify: Your example text here
Your message contains: offensive speech
```

---

## Project Structure
```plaintext
├── data_process.py       # Handles dataset loading and preprocessing.
├── model_create.py       # Defines the model architecture.
├── model_.py             # Prepares and trains the model.
├── predict.py            # Predicts the category of a given text.
├── try.py                # Main script for training and prediction.
```

---

## Dataset
The model is trained on the **hate-offensive-speech** dataset from the [Hugging Face Datasets](https://huggingface.co/datasets). This dataset includes tweets labeled as:
- **0**: Hate speech
- **1**: Offensive speech
- **2**: Normal speech

---

## Model Details
The classifier uses a bidirectional LSTM with the following layers:
1. **Embedding Layer**: Converts words into dense vectors of fixed size.
2. **Bidirectional LSTM**: Captures sequential dependencies in text.
3. **Dense Layers**: Fully connected layers for classification.
4. **Dropout Layer**: Prevents overfitting.
5. **Output Layer**: Uses softmax activation for multiclass classification.

The model is compiled with:
- Loss function: `sparse_categorical_crossentropy`
- Optimizer: `adam`
- Metrics: `accuracy`

