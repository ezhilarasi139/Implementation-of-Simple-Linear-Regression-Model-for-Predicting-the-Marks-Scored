# EX NO : 2  Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import the needed packages.

2.Assigning hours to x and scores to y.

3.Plot the scatter plot.

4.Use mse,rmse,mae formula to find the values.
 
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: EZHILARASI N
RegisterNumber: 212224040088
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('student_scores.csv')

#displaying the content in  datafile
df.head()

df.tail()

#Segregating data to variables
X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

#splitting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
Y_pred

#displaying actual values
Y_test

print("NAME : EZHILARASI N")
print("REGISTER : 212224040088")
#graph plot for training data
plt.scatter(X_train,Y_train,color="blue")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

print("NAME : EZHILARASI N")
print("REGISTER : 212224040088")
#graph plot for test data
plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,regressor.predict(X_test),color="red")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Calculate Mean Absolute Error (MAE) and Mean Squared Error(MSE)
mse=mean_squared_error(Y_test,Y_pred)
print('MSE= ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)

```

## Output:
To Read Head and Tail Files

<img width="287" height="252" alt="image" src="https://github.com/user-attachments/assets/c59148fc-8c95-4bca-9373-becf859c323f" />





<img width="333" height="261" alt="image" src="https://github.com/user-attachments/assets/3bc9a389-5f3e-469f-bcbf-fb5bfed83c5d" />




Segregating data to variables

<img width="359" height="666" alt="image" src="https://github.com/user-attachments/assets/1a37e25c-7c7e-4ab2-8507-7597652f5127" />

<img width="1032" height="72" alt="image" src="https://github.com/user-attachments/assets/227ea895-4734-4e9a-ac0b-a30264f5703e" />


displaying predicted values
<img width="926" height="75" alt="image" src="https://github.com/user-attachments/assets/aa431d2d-b0cc-45c3-8602-edf424341a61" />

displaying actual values


<img width="742" height="57" alt="image" src="https://github.com/user-attachments/assets/df03c1bf-3a64-4bf2-b17c-0d5e95c3d387" />


graph plot for training data


<img width="926" height="617" alt="image" src="https://github.com/user-attachments/assets/564222b6-5b6c-411e-854e-922113671b9f" />


graph plot for test data


<img width="858" height="632" alt="image" src="https://github.com/user-attachments/assets/b1c36307-c8bb-4ee2-a196-b0613bd41ac6" />

Calculate Mean Absolute Error (MAE) and Mean Squared Error(MSE)


<img width="295" height="87" alt="image" src="https://github.com/user-attachments/assets/59e9b40a-ad0e-47dd-8cce-cb9e31ef651c" />







## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
