import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd

credit_data = pd.read_csv('credit_data.csv')
print(credit_data.head())
print(credit_data.describe())
print(credit_data.corr())

feature = credit_data[['income', 'age', 'loan']]
target = credit_data['default']

# 30% of the dataset will be used for test, 70% of the dataset will be used for training
feature_train, feature_test, target_train, target_test = train_test_split(feature, target, test_size=0.3)
model = LogisticRegression()
model.fit = model.fit(feature_train, target_train)
predictions =  model.fit.predict(feature_test)

print("Confusion : ", confusion_matrix(target_test, predictions))
print("Accuracy Score : ", accuracy_score(target_test, predictions))