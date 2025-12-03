# Pokemon Classification

A PyTorch-based image classification project that classifies Pokemon images using a TinyVGG convolutional neural network architecture.

## Dataset

The dataset is sourced from [Kaggle - Pokemon Classification](https://www.kaggle.com/datasets/lantian773030/pokemonclassification) and contains images of 151 different Pokemon species organized into training and testing sets.

## Project Structure

This project follows the modular PyTorch structure outlined in [PyTorch Going Modular](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/05_pytorch_going_modular.md):

```
Pokemon_Classification/
├── data/
│   ├── train/          # Training images organized by Pokemon name
│   └── test/           # Testing images organized by Pokemon name (pulled randomly from original images)
├── models/             # Saved model files
├── src/
│   ├── data_setup.py   # DataLoader creation functionality
│   ├── engine.py       # Training and testing loops
│   ├── model_builder.py # TinyVGG model architecture
│   ├── train.py        # Main training script
│   ├── utils.py        # Helper functions
│   └── get_data.py     # Data downloading utilities
└── README.md
```

## Model Architecture

The project uses a TinyVGG architecture, a simplified version of the VGG network designed for educational purposes. The model includes:
- Convolutional blocks with ReLU activation
- Max pooling layers
- Fully connected classifier layers
- Supports 151 Pokemon classes

## Usage

Run the training script:
```bash
cd src
python train.py
```

The model will train for 20 epochs with the following default hyperparameters:
- Batch size: 32
- Learning rate: 0.001
- Hidden units: 64
- Image size: 64x64 pixels

## Requirements

- PyTorch
- torchvision
- Python 3.x
