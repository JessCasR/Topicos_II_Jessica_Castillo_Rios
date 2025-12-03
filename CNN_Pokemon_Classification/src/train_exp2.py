"""
Trains a PyTorch image classification model using device-agnostic code and ADAMW
"""

import os
import torch
import data_setup, engine, model_builder, utils
from torchvision import transforms
import pandas as pd

def main():

    NUM_EPOCHS   = 20
    BATCH_SIZE   = 32
    HIDDEN_UNITS = 64
    LEARNING_RATE = 0.0005
    IMAGE_SIZE   = 64
    WEIGHT_DECAY = 1e-4      # regularización L2

    train_dir = "../data/train"
    test_dir  = "../data/test"

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Used device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])

    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=train_transform,
        batch_size=BATCH_SIZE,
        test_transform=test_transform
    )

    model = model_builder.TinyVGG(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)
    ).to(device)

    # Loss y Optimizador ADAMW
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

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
    results_file = "exp2_results.pth"
    torch.save(results, f"../results/{results_file}")
    print(f"Resultados EXP2 guardados en ../results/{results_file}")

    # Exportar CSV
    csv_path = "../results/exp2_results.csv" 
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index_label="epoch")

    utils.save_model(
        model=model,
        target_dir="../models",
        model_name="pokemon_tinyvgg_exp2_adamw.pth"
    )

if __name__ == "__main__":
    main()
