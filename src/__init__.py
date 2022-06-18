"""
The __init__.py file indicates that the files in a folder are a part of a python package. Without an __init__.py file,
files cannot be imported from another directory.

An init file can be blank. It is essneitally the constructor of the package.
"""
# from .normalizer import *
# from .visualize_data import *
# from .NSEPred import *

import sys
sys.path.append("/home/agastya123/PycharmProjects/DeepLearning/NSE_Prediction/src/__init__.py")
sys.path.append("/home/agastya123/PycharmProjects/DeepLearning/NSE_Prediction/src/NSEPred.py")
sys.path.append("/home/agastya123/PycharmProjects/DeepLearning/NSE_Prediction/src/normalizer.py")
sys.path.append("/home/agastya123/PycharmProjects/DeepLearning/NSE_Prediction/src/visualize_data.py")

from .visualize_data import *
from .NSEPred import *
from .normalizer import *