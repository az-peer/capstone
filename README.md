## Undergraduate Capstone UCSB × SLAC

<p float="left" align="center">
  <img src="https://github.com/user-attachments/assets/7bf5d7a6-4883-46d2-935f-effc0a552041" width="350" height="250" style="margin-right: 40px; object-fit: cover;" />
  <img src="https://www.mercurynews.com/wp-content/uploads/2018/02/sjm-l-slac-0215-09.jpg?w=725" width="350" height="250" style="object-fit: cover;" />
</p>

## Collaborators  
Azfal Peermohammed, Haoran Yan, Jade Thai, Chris Zhao, David Lin, Kylie Maeda, Sichen Zhong

## About  
Protein X-ray crystallography is a powerful technique in structural biology that allows researchers to determine the atomic-level architecture of proteins. In this process, protein crystals are bombarded with high-intensity X-rays, producing two-dimensional diffraction patterns that encode the underlying molecular structure. These images are critical for advancing research in drug development, bioengineering, and energy science.

In collaboration with SLAC National Accelerator Laboratory, our capstone project leverages artificial intelligence to automate and enhance the analysis of these diffraction images. Specifically, we aim to develop deep learning models capable of interpreting geometric properties within the data—facilitating more accurate, efficient, and scalable structural elucidation.

<p align="center" style="margin-top: 40px;">
  <img src="https://proteindiffraction.org/media/datasets/FRS007_01_1_00001_eTCnsAP.300x300.png" width="300" height="300" style="object-fit: cover;" />
</p>

## Key Objectives

- **Group 1**:
  - Develop supervised deep learning models to analyze protein crystal diffraction images.
  - Accurately predict the geometric center of diffraction patterns.
  - Detect potential misalignments in the experimental setup.

- **Group 2**:
  - Construct neural networks to estimate the beam tilt angles that generated specific diffraction patterns.
  - Enable inverse modeling of the physical setup based on image features.

## Data  

- The dataset comprises protein diffraction images in two primary formats:
  - **Grayscale**: intensity-encoded representations of X-ray hits.
  - **Binary**: thresholded variants highlighting diffraction peak locations.
- Extended data includes sequential image frames simulating angular rotations of a crystal sample (video-format stretch goal).

## Methodologies  

- Both groups implemented and tested a variety of deep residual neural network (ResNet) architectures.
- Training was conducted using UCSB’s High-Performance Computing Cluster, POD, provided by the Center for Scientific Computing.
- Models were optimized for both regression and classification tasks, depending on the prediction target (e.g., center coordinates or tilt angle).
