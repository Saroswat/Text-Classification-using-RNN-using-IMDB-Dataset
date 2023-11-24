# Text Classification Using RNN on IMDB Dataset

This text classification tutorial demonstrates the implementation of a Recurrent Neural Network (RNN) on the IMDB large movie review dataset for sentiment analysis. The dataset comprises movie reviews labeled as either positive or negative sentiment.

## Purpose

The code showcases:
- Setup and initialization using TensorFlow and TensorFlow Datasets (TFDS).
- Preprocessing of the IMDB dataset for binary sentiment classification.
- Building an RNN-based model using TensorFlow/Keras for sentiment analysis.
- Model training, evaluation, and visualization of training metrics.

## Setup

This code requires TensorFlow and TensorFlow Datasets. Use the provided setup to install the necessary packages.

## Input Pipeline

The dataset is split into training and test sets and processed using TensorFlow Datasets. The code demonstrates:
- Dataset loading with `tfds.load`.
- Shuffle and batch setup for training and test datasets.
- Visualization of text and label pairs.

## Text Encoding

The raw text from the dataset is preprocessed using the `TextVectorization` layer. This layer adapts to the text and encodes it into indices for model input. The process involves setting vocabulary size, encoding text to indices, and reversing the encoding.

## Model Architecture

The model architecture consists of the following layers:
- `TextVectorization` layer for encoding text.
- `Embedding` layer for word representation.
- Bidirectional LSTM layer for sequence processing.
- Dense layers for final classification.

## Training and Evaluation

The code compiles and trains the model using a binary cross-entropy loss function and Adam optimizer. It tracks training accuracy, loss, and evaluates model performance on the test set.

## Additional Techniques

Demonstrates the use of stacking multiple LSTM layers in the model architecture for improved performance. It visualizes training metrics using Matplotlib.

## Usage

1. **Setup**: Install required packages.
2. **Execution**: Run code blocks sequentially to observe the training process and model evaluation.
3. **Model Customization**: Explore changing the model architecture or hyperparameters for different results.
4. **Visualizations**: Analyze training and validation metric plots to understand model performance.

## Sample Predictions

The code includes examples of predicting sentiment for custom input sentences using the trained model.
