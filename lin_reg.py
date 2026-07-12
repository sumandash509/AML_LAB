import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Dataset
X = np.array([1, 2, 3, 4]).reshape(-1, 1)
Y = np.array([2, 3, 5, 4])

# Create Linear Regression Model
model = LinearRegression()

# Train the model
model.fit(X, Y)

# Predict values
Y_pred = model.predict(X)

# Regression Line
slope = model.coef_[0]
intercept = model.intercept_

print("Regression Equation:")
print(f"Y = {intercept:.2f} + {slope:.2f}X")

# Evaluation Metrics
mae = mean_absolute_error(Y, Y_pred)
mse = mean_squared_error(Y, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y, Y_pred)

print("\nPredicted Values:")
print(Y_pred)

print("\nEvaluation Metrics")
print("MAE =", round(mae, 3))
print("MSE =", round(mse, 3))
print("RMSE =", round(rmse, 3))
print("R-squared =", round(r2, 3))

# Draw Regression Line
plt.scatter(X, Y, color='blue', label='Actual Data')
plt.plot(X, Y_pred, color='red', linewidth=2, label='Regression Line')

plt.title("Linear Regression using Least Squares Method")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)

plt.show()