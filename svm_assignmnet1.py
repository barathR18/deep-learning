
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import pandas as pd


#1. Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Filter to include only 'setosa' and 'versicolor' (labels 0 and 1)
filter_mask = (y == 0) | (y == 1)
X_filtered = X[filter_mask]
y_filtered = y[filter_mask]

# 2: Find out how many cases are there. Length of X
num_cases = len(X_filtered)
print("length of x:",num_cases)
#  3: find first 10 cases :
first_10_X = X_filtered[:10]
first_10_y = y_filtered[:10]
print("first 10 values of x are \n :",first_10_X)
print("first 10 values of y are:",first_10_y)

#  4: Find what attributes are there for X
attributes = iris.feature_names
print("attributes are:",attributes)

#  5: Find the mean and standard deviation of each attribute
means = np.mean(X_filtered, axis=0)
print("mean:",means)
stds = np.std(X_filtered, axis=0)
print("standard_deviation are:",stds)
attribute_stats = pd.DataFrame({'Mean': means, 'Standard Deviation': stds}, index=attributes)

# Step 6: Use SVC (kernel='linear') to train it
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.1, random_state=42)

# Train the SVC with a linear kernel
svc_linear = SVC(kernel='linear')
svc_linear.fit(X_train, y_train)


accuracy_linear = svc_linear.score(X_test, y_test)
print("accuracy rate of SVM with kernal linear model is :",accuracy_linear)

# Step 7: Test with a few sample data
#Create a few sample data points
sample_data = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [7.0, 3.2, 4.7, 1.4]
])


sample_predictions = svc_linear.predict(sample_data)
print("sample predications are:",sample_predictions)

# 8: Improve the classifier to get 95% accuracy
# Train the SVC with an RBF kernel
svc_rbf = SVC(kernel='rbf')
svc_rbf.fit(X_train, y_train)


accuracy_rbf = svc_rbf.score(X_test, y_test)

num_cases, first_10_X, first_10_y, attributes, attribute_stats, accuracy_linear, sample_predictions, accuracy_rbf

svc_rbf = SVC(kernel='rbf')
svc_rbf.fit(X_train, y_train)


accuracy_rbf = svc_rbf.score(X_test, y_test)
print("accuracy rate of SVM model using rbf model:",accuracy_rbf)
