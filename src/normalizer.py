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

    def __init__(self, option):
        super().__init__()
        self.option = option

    def testfunction(self):
        print(self.option)


if __name__ == "__main__":
    testobject = Normalize(option = None)





