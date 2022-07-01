"""
Initialization of the "models" package
"""


import sys
sys.path.append("/home/agastya123/PycharmProjects/DeepLearning/NSE_Prediction/models/__init__.py")
sys.path.append("/NSE_Prediction/models/CNN.py")
sys.path.append("/NSE_Prediction/models/MLP.py")
sys.path.append("/NSE_Prediction/models/LSTM.py")


from .CNN import *
from .LSTM import *
from .MLP import *