import numpy as np
from sklearn.datasets import fetch_california_housing   # 1.fetch the california_housing dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
housing = fetch_california_housing()
X = housing.data
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline = Pipeline([                                      # create pipeline standarize for input.
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(hidden_layer_sizes=[50,50,50],   # 2.With 3 hidden_layers with hidden_layer_sizes=[50, 50, 50]
                         activation='relu',    #activation as relu
                         solver='adam', # gradient descent as "adam"
                         alpha=0.0001,     # regulazartion as L2
                         random_state=42))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred,squared=False)
print(f'Mean Squared Error: {mse:.4f}')    # mean squARE error value.
predict= pipeline.predict(X_test)
print(predict)
