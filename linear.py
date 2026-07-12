

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X = [[1], [2], [3], [4]]
Y = [2, 3, 5, 4]

model = LinearRegression()
model.fit(X, Y)

pred = model.predict(X)

print("Slope =", model.coef_[0])
print("Intercept =", model.intercept_)

print("MAE =", mean_absolute_error(Y, pred))
print("MSE =", mean_squared_error(Y, pred))
print("RMSE =", np.sqrt(mean_squared_error(Y, pred)))
print("R2 Score =", r2_score(Y, pred))