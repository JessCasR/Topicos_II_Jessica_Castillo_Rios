"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import data_setup, engine, model_builder, utils
from torchvision import transforms
import pandas as pd

def main():

    # Setup hyperparameters
    NUM_EPOCHS = 20
    BATCH_SIZE = 32
    HIDDEN_UNITS = 64
    LEARNING_RATE = 0.001
    IMAGE_SIZE = 64

    # Setup directories
    train_dir = "../data/train"
    test_dir = "../data/test"

    # Setup target device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f'Used device: {device}')

    # Augmentación Suave
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])

    # Clean Test
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])

    # Create DataLoaders with help from data_setup.py
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=train_transform,     # AUG en train
        batch_size=BATCH_SIZE,
        test_transform=test_transform  # No AUG en test
    )

    # Create model with help from model_builder.py
    model = model_builder.TinyVGG(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)
    ).to(device)

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # Train
    results = engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=NUM_EPOCHS,
        device=device
    )

    os.makedirs("../results", exist_ok=True)
    results_file = "exp1_results.pth"
    torch.save(results, f"../results/{results_file}")
    print(f"Resultados EXP1 guardados en ../results/{results_file}")

    # Exportar CSV
    csv_path = "../results/exp1_results.csv" 
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index_label="epoch")

    print(f"Resultados formato CSV en: {csv_path}")

    utils.save_model(
        model=model,
        target_dir="../models",
        model_name="pokemon_tinyvgg_exp1_aug_suave.pth"
    )

if __name__ == '__main__':
    main()
