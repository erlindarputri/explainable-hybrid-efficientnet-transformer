# explainable-hybrid-efficientnet-transformer

Official implementation of the proposed Explainable Hybrid SE-Enhanced EfficientNet-B0–Transformer Model for Multi-Class Brain Tumor Classification.

## Description

This repository contains the implementation code for a hybrid model for brain tumor classification based on MRI images. The model combines EfficientNet-B0, the Squeeze-and-Excitation (SE) mechanism, and the Transformer.

## Classification Classes

The model is used to classify MRI images into four classes:
- Glioma
- Meningioma
- Pituitary
- No Tumor

## Model Architecture

1. **EfficientNet-B0:** Extracts visual features from MRI images.
2. **Squeeze-and-Excitation (SE):** Re-weights feature channels to emphasize relevant information.
3. **Transformer Encoder:** Learns inter-feature relationships to capture contextual information.
4. **Classification Head:** Generates predictions across the four classes.
5. **Grad-CAM and Grad-CAM++:** Visualizes the image regions that contribute to the model’s prediction results.

## Dataset

This study uses the Brain Tumor MRI dataset from Kaggle, which consists of 7,200 images across four classes.

## Implementation

This implementation covers the training, evaluation, and classification processes, as well as the visualization of model interpretation results using Grad-CAM and Grad-CAM++.
