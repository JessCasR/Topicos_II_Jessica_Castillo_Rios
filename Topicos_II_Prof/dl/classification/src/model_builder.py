import torch
from torch import nn

class ModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        # 2. Creamos las capas de entrada (lineales) capaces de manejar los features de entrada y clase de salida
        self.layer_1 = nn.Linear(in_features=10, out_features=20) # toma 10 features (X), produce 20 features
        self.layer_2 = nn.Linear(in_features=20, out_features=1) # toma 20 features, produce 1 feature (y)
    
    # 3. Definimos un método para la propagación (forward)
    def forward(self, x):
        # Regresa la capa de salida de layer_2, un solo features, con el mismo shape que y
        # El calculo pasa sobre layer_1 y luego su output es el input de layer_2
        return self.layer_2(self.layer_1(x)) 
    
class ModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=10, out_features=20)
        self.layer_2 = nn.Linear(in_features=20, out_features=20) # capa extra
        self.layer_3 = nn.Linear(in_features=20, out_features=1)
        
    def forward(self, x): 
        # z = self.layer_1(x)
        # z = self.layer_2(z)
        # z = self.layer_3(z)
        # return z
        return self.layer_3(self.layer_2(self.layer_1(x)))
    
class ModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=10, out_features=20)
        self.layer_2 = nn.Linear(in_features=20, out_features=20)
        self.layer_3 = nn.Linear(in_features=20, out_features=1)
        self.relu = nn.ReLU() # <- Se añade función de activación ReLU
        # También se puede usar sigmoid, pero se tendría que quitar en la parte de transformación del output 
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
      # ReLU se aplica entre capas
       return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))