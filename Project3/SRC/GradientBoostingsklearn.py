import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sklearn import metrics


data = pd.read_csv("processed.switzerland.data")
data.info()
