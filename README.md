# MLOps_Jan2025


### Overall goal of the project
The main goal of this project is to develop a tool for bird species classification. The tool will feature a user interface where users can upload bird images and receive predictions of the bird's species. Along with the classification, the system will also provide a confidence score that reflects the certainty of the classification. We strive to also include an explainability component, leveraging saliency maps and part annotations to highlight the specific parts of the bird that influenced the classification decision.

### What framework are you going to use and do you intend to include the framework into your project?
We will use the PyTorch Image Models (TIMM) as our PyTorch framework for creating the model based on a pre-trained ResNet50 model.

### What data are you going to run on (initially, may change)?
CUB200 dataset. Consists of images of birds in the wild, annotated by species and feature tagging, indicating colors, shapes etc of different parts of the birds. The Dataset includes 200 different species. The dataset has 11788 images.

### What models do you expect to use?
We expect to use a CNN model from the huggingface library pytorch-image-model (timm), which is a PyTorch based computer vision package. Most likely ResNet50. We won’t use a pretrained model, as our dataset claims, that there is an overlap between its images and the ones ResNet is pretrained on from ImageNet.
We might also look into other models, like ResNet101, VGG or DenseNet if our model doesn’t perform sufficiently.

## Project Structure

```plaintext
MLOps_Jan2025/
│
├── docker/                   # Directory for Docker-related files
│   ├── Dockerfile            # Dockerfile for building the container
│
├── data/                     # Store datasets and data-related files
│   └── folders containing images of birds         # Bird dataset
│
├── src/                      # Source code
│   ├── __init__.py           # Makes src a package
│   ├── config.py             # Configuration settings
│   ├── datasets.py           # Dataset and data loading utilities
│   ├── model.py              # Model definitions
│   ├── train.py              # Training logic
│   ├── test.py               # Testing and evaluation logic
│   ├── utils.py              # Utility functions
│   └── app.py                # Main entry point for applications (e.g., API)
│
├── figures
│   └── train_val_losses.png
│
├── .gitignore                # Git ignore file
├── README.md                 # Project documentation
├── requirements.txt          # List of Python dependencies
└──  model.pth                # Saved model file
