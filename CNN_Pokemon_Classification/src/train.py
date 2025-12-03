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

    # Create transforms
    data_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(25),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Create DataLoaders with help from data_setup.py
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=BATCH_SIZE
    )

    # Create model with help from model_builder.py
    model = model_builder.TinyVGG(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)
    ).to(device)

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE)
    # optimizer = torch.optim.SGD(model.parameters(),
    #                             lr=LEARNING_RATE)

    # Start training with help from engine.py
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
    results_file = "exp0_results.pth" 
    torch.save(results, f"../results/{results_file}")
    print(f"Resultados guardados en ../results/{results_file}")

    csv_path = "../results/exp0_results.csv" 
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index_label="epoch")

    print(f"Resultados formato CSV en: {csv_path}")

    # Save the model with help from utils.py
    utils.save_model(model=model,
                     target_dir="../models",
                     model_name="pokemon_tinyvgg_v0.pth")

if __name__ == '__main__':
    main()