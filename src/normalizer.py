"""
Class to Normalize and scale  the data
"""


import numpy as np
import torch
from torch import nn
import pandas as pd

class Normalize(nn.Module):
    """
    Class to normalize the data.
    """

    def __init__(self, data):
        super().__init__()
        self.data = data


    def fit_transform(self):

        """
        Function to normalize the data given to it: input can be a Tensor, DataFrame or Array
        :return: normalized data
        """
        if isinstance(self.data, pd.DataFrame):
            stock_data = self.data.iloc[:,1:]
            stock_data = torch.from_numpy(np.array(stock_data))

        else:
            stock_data = self.data[:,1:]
            stock_data = torch.from_numpy(np.array(stock_data))

        return stock_data






if __name__ == "__main__":

    """-----------------------------------------------------------------------------------------------------------------
    Testing the functions in the class Normalize
    -----------------------------------------------------------------------------------------------------------------"""
    testdata = np.random.randn(5,6)
    test_dataframe = pd.DataFrame(testdata)
    testobject = Normalize(data = pd.DataFrame(np.random.randn(5,6)))





