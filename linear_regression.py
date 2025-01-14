# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Generate synthetic data
# X: Features (independent variable)
# y: Target (dependent variable)
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # 100 data points, 1 feature
y = 4 + 3 * X + np.random.randn(100, 1)  # Linear relation: y = 4 + 3*X + noise

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print model parameters
print(f"Intercept (b0): {model.intercept_[0]:.2f}")
print(f"Coefficient (b1): {model.coef_[0][0]:.2f}")

# Visualize the results
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_test, y_pred, color='red', label='Regression Line')
plt.xlabel("X (Feature)")
plt.ylabel("y (Target)")
plt.legend()
plt.show()
