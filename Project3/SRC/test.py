import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

"""Collecting data with pandas"""
df = pd.read_csv("heart.csv")
df.info()
X = df.drop(["HeartDisease"], axis=1) #Everything but "HeartDisease"
y = df["HeartDisease"] #All data

"""Lable Encoding of non-numerical columns """
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df1 = df.copy()
df1['Sex']=le.fit_transform(df1['Sex'])
df1['RestingECG']=le.fit_transform(df1['RestingECG'])
df1['ChestPainType']=le.fit_transform(df1['ChestPainType'])
df1['ExerciseAngina']=le.fit_transform(df1['ExerciseAngina'])
df1['ST_Slope']=le.fit_transform(df1['ST_Slope'])

data_le = df1.drop('HeartDisease', axis=1) #removes column HeartDisease

X_train_le, X_test_le, y_train_le, y_test_le = train_test_split(data_le, y, test_size=0.2)

degree = np.array([2, 3, 4, 5, 6, 7, 8])

for i in degree:
  print("Degree = ", i)
  clf = SVC(degree = i, kernel = "poly", gamma='auto')
  clf.fit(X_train_le, y_train_le)
  print("Accuracy:", clf.score(X_test_le, y_test_le))
