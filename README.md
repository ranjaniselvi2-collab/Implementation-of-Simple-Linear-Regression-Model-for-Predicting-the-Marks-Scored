# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm


1. **Collect Data**

   Take input values: hours studied (X) and marks scored (Y).

2. **Calculate Mean Values**

    Find mean of X and mean of Y.

3. **Compute Slope and Intercept**

    Calculate slope (m) and intercept (b) using formulas.

4. **Form Linear Equation**

    Create regression equation: ( Y = mX + b ).

5. **Predict Marks**

    Use the equation to predict marks for new input values.


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Ranjani S
RegisterNumber:  212225230224
*/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks_Scored":  [35, 40, 50, 55, 60, 65, 70, 80, 85, 95]
}
df = pd.DataFrame(data)


print("Dataset:\n", df.head())
df
X = df[["Hours_Studied"]]  
y = df["Marks_Scored"] 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nModel Parameters:")
print("Intercept (b0):", model.intercept_)
print("Slope (b1):", model.coef_[0])

print("\nEvaluation Metrics:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
plt.figure(figsize=(8,6))
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, model.predict(X), color='red', linewidth=2, label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Simple Linear Regression: Predicting Marks")
plt.legend()
plt.grid(True)
plt.show()
```

## Output:

<img width="1064" height="806" alt="Screenshot 2026-04-24 144808" src="https://github.com/user-attachments/assets/bdaeade0-3856-4560-a154-247ffd23cc3a" />
<img width="1069" height="761" alt="Screenshot 2026-04-25 100641" src="https://github.com/user-attachments/assets/5a744167-f1d7-4ff6-bde1-c4122a909e53" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
