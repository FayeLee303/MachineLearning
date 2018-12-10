"""PLA算法"""
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

if __name__ == "__main__":
    iris = load_iris()
    df = pd.DataFrame(iris.data,columns=iris.feature_names)
    df['label'] = iris.target()