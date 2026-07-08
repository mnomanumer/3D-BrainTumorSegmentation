# 3D Brain Tumor Segmentation & Volumetric Analytics

An advanced deep learning framework engineered for high-dimensional semantic segmentation of multi-modal brain structures using volumetric medical imaging.

## The Core Uniqueness
Transitions completely away from basic 2D image classification into high-dimensional 3D semantic segmentation. The backend architecture processes raw NIfTI files and computes real-time volumetric voxel counts, enabling clinical tracking of actual tumor reduction or growth across temporal scans.

## Tech Stack & Requirements
* Deep Learning Framework: PyTorch
* Architectural Pattern: 3D U-Net
* Medical Image Processing: Nibabel (NIfTI format manipulation)
* Loss Optimization: Dice Loss Function
* Deployment Interface: Gradio UI hosted via Google Colab (T4 GPU)

## Storage & Hardware Walkaround
Built to circumvent strict local storage constraints (e.g., small 128GB SSD limits). The ingestion pipeline pulls medical image tensors directly into Google Colab via the Kaggle API, performs distributed GPU training on a cloud-allocated T4 instance, and automatically streams checkpoint weights straight to Google Drive.