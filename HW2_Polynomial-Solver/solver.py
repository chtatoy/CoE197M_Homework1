# Camille Marie H. Tatoy
# 2015-11050
# CoE197M-THY

# Homework 2:
from tinygrad.tensor import Tensor
import tinygrad.nn.optim as optim
# import csv
import pandas as pd

# Use data_train.csv to estimate the degree and coefficients of a polynomial
# a = open('data_train.csv','r')
# data_train = csv.reader(a)

# Use data_test.csv to test the generalization of the learned function
# b = open('data_test.csv','r')
# data_test = csv.reader(b)

# --------------------------------------------------

class TinyBobNet:
  def __init__(self):
    self.l1 = Tensor.uniform(784, 128)
    self.l2 = Tensor.uniform(128, 10)

  def forward(self, x):
    return x.dot(self.l1).relu().dot(self.l2).logsoftmax()

model = TinyBobNet()
optim = optim.SGD([model.l1, model.l2], lr=0.001)

df = pd.read_csv('data_train.csv')
x = df["x"]
y = df["y"]

out = model.forward(x)
loss = out.mul(y).mean()
optim.zero_grad()
loss.backward()
optim.step()

# --------------------------------------------------

# a.close()
# b.close()