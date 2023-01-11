import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# TODO 1: roda, mexe um pouco, tenta ver a lógica
# TODO 2: faz o exemplo do site pytorch mesmo, pra imagens, mas usando esses dados de 1,2 3,4 etc

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def test():
    data = torch.tensor([[1,2], [3,4], [5, 6], [7, 8], [9,10]])
    # Simplesmente multiplica por 2
    y = torch.tensor([[2,4], [6,8], [10, 12], [14, 16], [18,20]])

    #-------------------------------------
    # from https://machinelearningmastery.com/making-predictions-with-multilinear-regression-in-pytorch/ 

    pass

if __name__ == '__main__':
    test();