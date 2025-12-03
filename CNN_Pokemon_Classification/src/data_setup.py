"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = 4

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose,           # transform para TRAIN
    batch_size: int, 
    num_workers: int = NUM_WORKERS,
    test_transform: transforms.Compose = None  # transform para TEST (opcional)
):
    """
    Creates training and testing DataLoaders.

    Args:
      train_dir: Path to training directory.
      test_dir: Path to testing directory.
      transform: Transformations for TRAINING data (with augmentation).
      batch_size: Number of samples per batch.
      num_workers: Number of workers per DataLoader.
      test_transform: Transformations ONLY for TEST data (sin augmentations).

    Returns:
      train_dataloader, test_dataloader, class_names
    """

    # Si no se pasa un test_transform, usamos el mismo transform que train
    if test_transform is None:
        test_transform = transform

    # Crear datasets
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)

    # Obtener nombres de clases
    class_names = train_data.classes

    # DataLoader para TRAIN
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # DataLoader para TEST
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,  # test nunca se mezcla
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, class_names
