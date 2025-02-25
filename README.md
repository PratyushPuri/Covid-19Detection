# COVID-19 Detection from X-ray using Deep Learning

This repository contains a deep learning model for detecting COVID-19 from X-ray images. The model uses a customized sequential CNN architecture to classify X-ray images as either COVID-19 positive or negative.

## Overview

This project was developed as part of my bachelor's engineering final year. It demonstrates the application of deep learning techniques in medical image analysis, specifically for COVID-19 detection from chest X-rays.

## Model Architecture

The model is built using TensorFlow and Keras with a customized sequential architecture:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

## Dataset

- Approximately 3,000 X-ray images were used for training
- Images were preprocessed to 150x150 pixels
- Dataset can be found at: [GitHub Dataset Repository](https://github.com/PratyushPuri/datasets)

## Performance

The model achieves:
- Test Accuracy: 92.56%
- Test Loss: 0.2053

## Requirements

To run this project, you will need:
- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Google Colab or Jupyter Notebook

## Installation and Usage

1. Clone this repository:
   ```
   git clone https://github.com/PratyushPuri/Covid-19Detection.git
   ```

2. Download the dataset from: [GitHub Dataset Repository](https://github.com/PratyushPuri/datasets)

3. Open the notebook in Google Colab or Jupyter Notebook

4. Run the notebook cells sequentially to:
   - Load and preprocess the data
   - Build and train the model
   - Evaluate model performance
   - Make predictions on new X-ray images

## Code Structure

- `COVID19_Detection.ipynb`: Main notebook containing the model training and evaluation
- `data/`: Folder containing the dataset (you need to download this separately)
- `models/`: Saved model weights

## Limitations and Future Work

As this is a beginner-level project developed during my bachelor's engineering final year, there are several areas for improvement:

- Larger and more diverse dataset could improve generalization
- Advanced architectures like ResNet or EfficientNet could be explored
- Additional preprocessing techniques could be implemented
- Clinical validation would be required for any practical application

