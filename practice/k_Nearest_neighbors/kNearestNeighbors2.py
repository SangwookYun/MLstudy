import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

data = pd.read_csv("../../credit_data.csv")
print(data)

features = data[["income", "age", "loan"]]
target = data.default

# machine learning handle arrays not data-frames
X = np.array(features).reshape(-1, 3)
y = np.array(target)

# normalization
X = preprocessing.MinMaxScaler().fit_transform(X)

feature_train, feature_test, target_train, target_test = train_test_split(X, y, test_size = 0.3)
n_neighbors = 20
model = KNeighborsClassifier(n_neighbors=n_neighbors)
fitted_model = model.fit(feature_train, target_train)
predictions = fitted_model.predict(feature_test)
print("before cv")
print("n_neighbors : ", n_neighbors)
print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))

# to find n_neighbors, cross_validation is needed
cross_valid_scores = []
for k in range(1, 100):
    knn = KNeighborsClassifier(n_neighbors =k)
    scores = cross_val_score(knn, X, y, cv=10, scoring = "accuracy")
    cross_valid_scores.append(scores.mean())

print("Optimal k with cv : " , np.argmax(cross_valid_scores))

print("after cv")
print(np.argmax(cross_valid_scores)) 
model = KNeighborsClassifier(n_neighbors=np.argmax(cross_valid_scores))
fitted_model = model.fit(feature_train, target_train)
predictions = fitted_model.predict(feature_test)

print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))
