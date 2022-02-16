import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd

credit_data = pd.read_csv('credit_data.csv')
print(credit_data.head())
print(credit_data.describe())
print(credit_data.corr())

feature = credit_data[['income', 'age', 'loan']]
target = credit_data['default']

X = np.array(feature).reshape(-1, 3)
y = np.array(target)

model = LogisticRegression()
predicted = cross_validate(model, X, y, cv = 5)
print("Predicted : ", predicted['test_score'])
print(np.mean(predicted['test_score']))