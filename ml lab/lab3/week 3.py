#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import csv
data= pd.read_csv('/home/student/Desktop/ml lab_210962114/week 3/hepatitis_csv.csv')
data


# In[6]:


data.fillna(1)


# In[8]:


columns1 =['bilirubin']
data1 = data.drop(columns=columns1)
data1


# In[9]:


data2= pd.get_dummies(data , columns = ['age'] ,prefix= ['fatigue'])
data2


# In[12]:



missing_values = data.isna()

data_cleaned = data.dropna()

data_filled = data.fillna(0)
data_filled


# In[13]:


data_arrays=data.values
data_arrays


# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


file_path = '/home/student/Desktop/ml lab_210962114/week 3/diabetes_csv.csv' 
data = pd.read_csv(file_path)




X = data[['Glucose']]
y = data['Outcome']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()


model.fit(X_train, y_train)


intercept = model.intercept_
slope = model.coef_[0]


y_pred = model.predict(X_test)


rmse = np.sqrt(mean_squared_error(y_test, y_pred))


plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='yellow', label='Actual Data')
plt.plot(X_test, y_pred, color='green', label='Predicted')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title(f'Linear Regression (RMSE: {rmse:.2f})')
plt.show()


print("Intercept (B0):", intercept)
print("Slope (B1):", slope)
print("Root Mean Squared Error (RMSE):", rmse)


# In[16]:


data_cleaned1 = data.dropna()
data_cleaned1


# In[ ]:




