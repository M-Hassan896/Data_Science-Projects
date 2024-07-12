# Speech Enhancement System

## Introduction

Speech enhancement systems aim to improve the quality and intelligibility of speech signals that are corrupted by noise or other unwanted distortions. This project presents a speech enhancement system implemented using unsupervised learning techniques.

## Unsupervised Learning

Unsupervised learning refers to training models without explicit labels, allowing them to learn patterns and structures inherent in the data. The system utilizes a convolutional neural network (CNN) to denoise audio signals and enhance speech quality.

## System Overview

The speech enhancement system follows a pipeline that involves audio loading, spectrogram conversion, model training, and evaluation. The key components and techniques used in the system are as follows:

### Techniques

1. **Audio Loading**: 
   - The system reads clean and noisy audio files using the `load_wav(file_path)` function.
   - The audio files are stored as tensors.

2. **Spectrogram Conversion**: 
   - The `convert_to_spectrogram(wav)` function transforms the audio waveforms into magnitude spectrograms using the Short-Time Fourier Transform (STFT) technique.
   - This conversion provides a time-frequency representation of the audio signals.

3. **Unsupervised Learning**: 
   - The system employs unsupervised learning techniques, where the model is trained on clean and noisy spectrograms without explicit labels.
   - This allows the model to learn the underlying structures and patterns in the data.

4. **Convolutional Neural Network (CNN)**: 
   - The model architecture, built using the Keras library, consists of multiple convolutional layers with batch normalization.
   - CNNs are powerful for learning hierarchical representations from spectrograms and capturing relevant features for speech enhancement.

5. **Mean Squared Error (MSE) Loss**: 
   - The model is trained using the mean squared error loss function ('mse').
   - It measures the discrepancy between the denoised spectrograms predicted by the model and the corresponding clean spectrograms.
   - Minimizing this loss guides the model to learn denoising patterns.

6. **Adam Optimizer**: 
   - The Adam optimizer is used during model training with a learning rate of 0.001.
   - Adam adapts the learning rate based on gradient information, facilitating efficient convergence and better generalization.

7. **Model Checkpointing**: 
   - The `ModelCheckpoint` callback is employed to save the best-performing model based on validation loss.
   - This prevents overfitting and ensures that the model with the highest validation performance is saved.

8. **Short-Time Objective Intelligibility (STOI)**: 
   - The evaluation of the speech enhancement system is based on the STOI metric.
   - The `pystoi.stoi` function from the `pystoi` library is used to calculate the STOI score between the denoised speech and the original clean speech.
   - STOI provides a measure of speech intelligibility and allows for quantitative assessment of the system's performance.

