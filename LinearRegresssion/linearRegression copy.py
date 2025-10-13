import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Method 2: Sklearn (More robust)
from sklearn.preprocessing import OneHotEncoder
# read the data

data = pd.read_csv('Salary_Data.csv')
#print(data.head())

model = LinearRegression()

# #convert the string data to numerical data
# ohe = OneHotEncoder(sparse_output=False, drop='first')
# job_encoded = ohe.fit_transform(data[['Job Title']])
# data = pd.concat([data, pd.DataFrame(job_encoded, columns=ohe.get_feature_names_out(['Job Title']))], axis=1)
# data = data.drop('Job Title', axis=1)

X1 = data.iloc[:, 0].values
# X2 = data.iloc[:, 3].values
y1 = data.iloc[:, -1].values
#removed the missing values rows from the data set
data = data[~np.isnan(X1) & ~np.isnan(y1)]
# data = data[~np.isnan(X1) & ~np.isnan(y1) & ~np.isnan(X2)]

#X1 = data.iloc[:, [0, 3]].values  # Select columns 0 and 3 specifically
X1 = data.iloc[:, 0].values .reshape(-1, 1)
y1 = data.iloc[:, -1].values.reshape(-1, 1)

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=0)

model.fit(X1_train, y1_train)

y1_pred = model.predict(X1_test)

y3 = model.predict([[56]])
#plt.scatter(X1_train, y1_train, color='red')'
plt.scatter(X1_test, y1_test, color='blue')
print(y3)
#plt.scatter(X1_test[:,0], y1_test, color='blue')
plt.plot(X1_test, y1_pred, color='yellow')
# plot the single value

plt.scatter([[56]], y3, color='green')

plt.show()

