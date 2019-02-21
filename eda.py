
"""
The following code is for exploratory data analysis of
Housing Price Data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

df = pd.read_csv('./data/train.csv')
print('Number of Rows:', max(df.count()))
#Print Missing Rows for each column
print(df.isna().sum())


