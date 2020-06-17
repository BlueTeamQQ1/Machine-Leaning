import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

df =  pd.read_csv("Position_Salaries.csv")
X  = df.iloc[:,1:-1].values
Y  =df.iloc[:,-1].values.reshape(-1,1)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators = 10, random_state = 0)
model.fit(X, Y)

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.scatter(X, Y, color = "red")
plt.title("Position Level vs Salary")
plt.xlabel("Position Level")
plt.ylabel("Salary (dollars/year)")
X_dummy = np.arange(0, 10, 0.1).reshape(-1, 1)
Y_dummy_pred = model.predict(X_dummy)

plt.subplot(1,2,2)
plt.scatter(X, Y, color = "red")
plt.plot(X_dummy, Y_dummy_pred, color = "blue")
plt.title("Position Level vs Salary")
plt.xlabel("Position Level")
plt.ylabel("Salary (dollars/year)")
plt.show()