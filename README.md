# Linear Regression - How a Line is Taken  

## 📌 Introduction  
Linear Regression is a statistical method to model the relationship between an independent variable (\(x\)) and a dependent variable (\(y\)) using a straight line.  

The equation of a line is:  
\[
y = mx + c
\]  
where:  
- \( y \) = predicted output (dependent variable)  
- \( x \) = input feature (independent variable)  
- \( m \) = slope of the line  
- \( c \) = y-intercept  

---

## 🔢 How a Line is Taken?  

### 1️⃣ **Initialize a Random Line**  
At first, we assume a random line with some initial values of \( m \) and \( c \).  
Example:  
\[
y = 0.5x + 1
\]  

### 2️⃣ **Measure the Error (Cost Function - MSE)**  
To check how well the line fits, we calculate the **Mean Squared Error (MSE):**  
\[
MSE = \frac{1}{n} \sum (y_i - \hat{y}_i)^2
\]  
where \( y_i \) is the actual value and \( \hat{y}_i \) is the predicted value.

### 3️⃣ **Optimize the Line (Finding Best \( m \) and \( c \))**  
We adjust the values of \( m \) and \( c \) to minimize the error using:  

#### ✅ **Least Squares Method (Direct Calculation)**
\[
m = \frac{n \sum (x_i y_i) - \sum x_i \sum y_i}{n \sum x_i^2 - (\sum x_i)^2}
\]
\[
c = \frac{\sum y_i - m \sum x_i}{n}
\]

#### ✅ **Gradient Descent (Iterative Approach)**
\[
m = m - \alpha \frac{\partial MSE}{\partial m}
\]
\[
c = c - \alpha \frac{\partial MSE}{\partial c}
\]

where \( \alpha \) is the learning rate.

---

## 🚀 **Python Code for Linear Regression**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample Data
X = np.array([1, 2, 3, 4]).reshape(-1, 1)  # Independent variable (2D)
y = np.array([2, 3, 5, 7])  # Dependent variable (1D)

# Creating the model
model = LinearRegression()
model.fit(X, y)  # Train the model (Find best-fit line)

# Predict values
predicted_y = model.predict(X)

# Plot the data
plt.scatter(X, y, color='red', label="Actual Data")
plt.plot(X, predicted_y, color='blue', label="Best-Fit Line")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Linear Regression - Best Fit Line")
plt.show()
