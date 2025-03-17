# Linear Regression - Understanding the Best Fit Line

## Introduction  
Linear Regression is a fundamental machine learning algorithm used to model the relationship between a dependent variable (y) and one or more independent variables (x). The goal is to find the best-fit line that minimizes the error between predicted and actual values.

The equation of a line is:  
y = mx + c  

where:  
- y = predicted output (dependent variable)  
- x = input feature (independent variable)  
- m = slope of the line  
- c = y-intercept  

---

## How the Best-Fit Line is Determined  

### 1. Initializing a Random Line  
Initially, we assume a random line with some values for m and c.

### 2. Measuring the Error (Mean Squared Error - MSE)  
To check how well the line fits, we calculate the Mean Squared Error (MSE):

MSE = (1/n) * Σ(y_actual - y_predicted)^2  

where:  
- y_actual = actual value  
- y_predicted = predicted value  
- n = number of data points  

### 3. Optimizing the Line (Finding the Best m and c)  
To minimize the error, we adjust m and c using:  

#### Least Squares Method (Direct Calculation)  
m = [ n * Σ(xy) - Σx * Σy ] / [ n * Σx^2 - (Σx)^2 ]  
c = (Σy - m * Σx) / n  

#### Gradient Descent (Iterative Approach)  
m = m - α * (∂MSE/∂m)  
c = c - α * (∂MSE/∂c)  

where α is the learning rate.

---

## Python Code for Linear Regression
```python
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression

# Sample Data
X = np.array([1, 2, 3, 4]).reshape(-1, 1)  # Independent variable
y = np.array([2, 3, 5, 7])  # Dependent variable

# Creating the model
model = LinearRegression()
model.fit(X, y)  # Train the model

# Predict values
predicted_y = model.predict(X)

# Save the trained model using pickle
with open("linear_regression_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Load the model back
with open("linear_regression_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# Predict using the loaded model
new_data = np.array([[5]])  # Example input
predicted_value = loaded_model.predict(new_data)
print(f"Predicted Value for input 5: {predicted_value[0]}")

# Plot the data
plt.scatter(X, y, color='red', label="Actual Data")
plt.plot(X, predicted_y, color='blue', label="Best-Fit Line")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Linear Regression - Best Fit Line")
plt.show()

