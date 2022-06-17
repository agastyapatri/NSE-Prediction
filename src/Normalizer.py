"""
Normalizing the data
"""


import numpy as np
import torch
from torch import nn
import pandas as pd



class Normalizer(nn.Module):
    """
    defining the type of scaling that is occurring
    """
    def __init__(self, option):
        super().__init__()
        self.option = option


    def normalize(self, data):
        """
        Function to normalize and scale the data
        :param data: either a NumPy array, a PyTorch Tensor, or a Pandas Dataframe
        :return: normalized data
        """
        if self.option == "tensor":
            mean = torch.mean(data, dtype = torch.float32)
            std = torch.std(data)

            normalized_tensor = (data - mean) / std
            result = normalized_tensor

        elif self.option == "array":
            mean = np.mean(data, dtype=np.float32)
            std = np.std(data, dtype=np.float32)

            normalized_array = (data - mean) / std
            result = normalized_array

        elif self.option == "dataframe":
            mean = pd.mean(data, dtype=torch.float32)
            std = pd.std(data, dtype=torch.float32)

            normalized_dataframe = (data - mean) / std
            result = normalized_dataframe

        else:
            print("Not a Valid type of data; choose between tensor, array and dataframe")

        return result



if __name__ == "__main__":
    scaler = Normalizer(option="tensor")
    test = torch.randn(5,6)
    print(f"mean of the original tensor is : {torch.mean(test)}")
    normalized_test = scaler.normalize(test)
    print(f"mean of the normalized tensor is : {torch.mean(normalized_test)}")