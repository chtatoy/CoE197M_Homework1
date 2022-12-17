# Camille Marie H. Tatoy
# 2015-11050
# CoE197M-THY

# Homework 2:
from tinygrad.tensor import Tensor
import tinygrad.nn.optim as optim
# import csv
import pandas as pd
import matplotlib.pyplot as plt
# import numpy as np
from sympy import *


# Data Preprocessing

# Use data_train.csv to estimate the degree and coefficients of a polynomial
df = pd.read_csv('data_train.csv')
x_train = df["x"]
y_train = df["y"]

# Use data_test.csv to test the generalization of the learned function
df = pd.read_csv('data_test.csv')
x_test = df["x"]
y_test = df["y"]

# ---

# Fitting a Model

# Backward propagation
# Model using tinygrad.nn.optim.SGD
class TinyBobNet:
  def __init__(self):
    self.l1 = Tensor.uniform(784, 128)
    self.l2 = Tensor.uniform(128, 10)

  def forward(self, x):
    return x.dot(self.l1).relu().dot(self.l2).logsoftmax()

model = TinyBobNet()
optim = optim.SGD([model.l1, model.l2], lr=0.001)   # learning rate = 0.001

out = model.forward(x_train)
loss = out.mul(y_train).mean()
optim.zero_grad()
loss.backward()
optim.step()

# hypothesis function, h
# h_θ(x) = θ_0 + θ_1*x_1

# Predict Coefficient
train_pred = out.predict(x_train)
test_pred = out.predict(x_test)

diff1 = []
diff1 = x_train
diff2 = []
# Predict Degree
# an n-degree polynomial has at most n x-intercepts and n-1 turning pts
# to find degree of polynomial with data, find level where "difference" is constant
# while True:
#   for c in len(diff1):
#     if c == 0:
#       subt = diff1[c]
#     diff2[c-1] = diff1[c]-subt
#     subt = diff1[c]
#     i+=1
#   if diff1[0]==diff2[0]:
#     break
# degree = i

# degree of a function can be computed as deg f(x) = lim_x→∞ (x*f'(x))/f(x)
x = x_train
y = y_train
z = y.matmul(x).sum()
degree = limit(y.grad,x.grad,oo)

# ---

# Visualizing Results of Model

# Plot
plt.scatter(x_train, y_train, c='blue', label='train x')
plt.scatter(x_test, y_test, c='green', label='test x')
plt.scatter(x_train, train_pred, c='red', label='train pred')
plt.scatter(x_test, test_pred, c='orange', label='test pred')
plt.legend()
plt.show()

# ---

# a.close()
# b.close()

# --------------------------------------------------










# github.com/geohot/tinygrad
# stackoverflow.com/questions/70638278/training-a-neural-network-to-learn-polynomial-equation
# askpython.com/python/examples/polynomial-regression-in-python
# https://rickwierenga.com/blog/ml-fundamentals/polynomial-regression.html
