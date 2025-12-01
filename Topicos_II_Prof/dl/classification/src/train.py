"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import engine, model_builder

# Setup hyperparameters
NUM_EPOCHS = 1000
#HIDDEN_UNITS = 10
LEARNING_RATE = 0.01

# Setup directories
train_dir = "../data/train_clean.csv"
df = pd.read_csv(train_dir)
X_df = df.drop(columns="Survived")
y_df = df["Survived"] 
X_df_train, X_df_test, Y_df_train, Y_df_test = train_test_split(X_df,
                                                                y_df,
                                                                test_size=0.2,
                                                                random_state=42)

# to numpy                                                             
X_array_train = X_df_train.to_numpy()
X_array_test = X_df_test.to_numpy()
Y_array_train = Y_df_train.to_numpy()
Y_array_test = Y_df_test.to_numpy()

# to torch tensor
X_train = torch.from_numpy(X_array_train).type(torch.float)
X_test = torch.from_numpy(X_array_test).type(torch.float)
y_train = torch.from_numpy(Y_array_train).type(torch.float)
y_test = torch.from_numpy(Y_array_test).type(torch.float)

#test_dir = "data/pizza_steak_sushi/test"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else "cpu"


# Create model with help from model_builder.py
model = model_builder.ModelV2().to(device)
#model = model_builder.TinyVGG(
#    input_shape=3,
#    hidden_units=HIDDEN_UNITS,
#    output_shape=len(class_names)
#).to(device)

# Set loss and optimizer
#loss_fn = torch.nn.CrossEntropyLoss()
loss_fn = torch.nn.BCEWithLogitsLoss()
#optimizer = torch.optim.Adam(model.parameters(),
#                             lr=LEARNING_RATE)
optimizer = torch.optim.SGD(params=model.parameters(), 
                            lr=LEARNING_RATE)

# Start training with help from engine.py
engine.train(model=model,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             X_train=X_train,
             X_test=X_test,
             y_train=y_train,
             y_test=y_test,
             device=device)

# Save the model with help from utils.py
#utils.save_model(model=model,
#                 target_dir="models",
#                 model_name="05_going_modular_script_mode_tinyvgg_model.pth")