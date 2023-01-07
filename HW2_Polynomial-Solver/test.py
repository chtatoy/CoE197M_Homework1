import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. How to download the dataset
df_train = pd.read_csv('data_train.csv') # data frames
#df.head() #display first 5 rows

# 2. Understanding the dataset
#df.describe() #statistics

#df.dtypes
# 3. Cleaning the dataset

# replace NaN value with mean of data

# 4. Understanding the Problem Statement

# 

#plt.scatter(df['x'],df['y'])
#plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# conda install -c anaconda scikit-learn
# numpy=1.22

# 5. Deciding the model to be used
# 6. Deciding the libraries to be used
# 7. Studying the in-built methods to be used

#lm = LinearRegression()
x_train=df_train[['x']]
y_train=df_train[['y']]
#lm.fit(X,Y) # first parameter should be a table thats why double brackets

# Yhat = lm.predict(X)
# lm_score = lm.score(X,Y) # vary from 0-1 very bad-good model
#print(lm_score)
# 0.059872261294025675

# plt.plot(X,Yhat)
# plt.scatter(X,Y)
# plt.show()

# need to improve

df_test = pd.read_csv('data_test.csv')
x_test=df_test[['x']]
y_test=df_test[['y']]

lm = LinearRegression()
lm.fit(x_train,y_train)
Yhat = lm.predict(x_train)
lm_score=lm.score(x_train,y_train)
#print(lm_score)
# 0.544476073143375

a=lm.predict(x_test)
print(y_test)
#print(np.mean((a-y_test)**2))

# plt.plot(x_train,Yhat)
# plt.scatter(x_train,y_train)

# plt.title('Prediction')
# plt.xlabel('x')
# plt.ylabel('y')

# plt.show()

# 8. Inferring the result obtained
# 9. Plotting the result

