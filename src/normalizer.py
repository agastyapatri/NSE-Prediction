"""
Class to Normalize and scale  the data
"""


import numpy as np
import torch
from torch import nn

class Normalize(nn.Module):
    """
    Class to normalize the data.
    """

    def __init__(self, data):
        super().__init__()
        self.data = data

    def fit_transform(self,  testdata):

        """
        Function to normalize the data given to it: input can be a Tensor, DataFrame or Array
        :return: normalized data
        """

        stock_data = torch.from_numpy(np.array(testdata))
        mu = torch.mean(stock_data)
        sigma = torch.std(stock_data)

        scaled_data = (stock_data - mu) / sigma

        return scaled_data

    def testfunction(self):
        print(self.option)


if __name__ == "__main__":
    testdata = np.random.randn(5,6)
    testobject = Normalize(data = testdata)





