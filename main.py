from NSE_Prediction.src.Normalizer import Normalizer
import torch

scaler = Normalizer(option="tensor")

test = torch.randn(5,6)
normalized_test = scaler.normalize(test)
