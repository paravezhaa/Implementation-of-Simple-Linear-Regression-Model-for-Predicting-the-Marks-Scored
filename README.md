# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: PARAVEZHAA M
RegisterNumber:212225220070

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv("student_scores.csv")
print("First 10 rows:")
print(df.head(10))


plt.scatter(df['Hours'], df['Scores'])
plt.xlabel('Hours Studied')
plt.ylabel('Scores')
plt.show()


x = df[['Hours']]  
y = df['Scores']   


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


model = LinearRegression()
model.fit(X_train, y_train)


sample_pred = model.predict(X_test.iloc[0].values.reshape(1,1))
print(f"Predicted score for first test sample: {sample_pred[0]}")


plt.scatter(df['Hours'], df['Scores'])
plt.plot(x, model.predict(x), color='red') 
plt.xlabel('Hours Studied')
plt.ylabel('Scores')
plt.show()


y_pred = model.predict(X_test)
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

*/
```

## Output:
<img width="1497" height="782" alt="Screenshot 2026-02-04 102723" src="https://github.com/user-attachments/assets/ac02f3ae-21e1-4ed5-9117-2bfe1bec9795" />
<img width="1496" height="744" alt="Screenshot 2026-02-04 103837" src="https://github.com/user-attachments/assets/af55900c-8fbc-44d5-b968-fe84da48b71e" />




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
