import numpy as np

def ackley(X, a=20, b=0.2, c=2*np.pi):
  # https://www.sfu.ca/~ssurjano/Code/ackleyr.html
  d = len(X)
  sum1 = np.sum(X ** 2)
  sum2 = np.sum(np.cos(c * X.astype('float')))

  term1 = -a * np.exp(-b * np.sqrt(sum1/d))
  term2 = - np.exp(sum2/d)

  y = term1 + term2 + a + np.exp(1)
  return y

def rastrigin(X):
  # https://www.sfu.ca/~ssurjano/Code/rastrr.html
  d = len(X)
  sum = np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X.astype('float')))
  y = 10 * d + sum
  return y

def rosenbrock(X):
  # https://www.sfu.ca/~ssurjano/Code/rosenr.html
  X = X.astype('float')
  d = len(X)
  xi = X[1:d-1]
  xnext = X[2:d]
  sum = np.sum(100 * (xnext - xi ** 2) ** 2 + (xi-1) ** 2 )
  return sum

# def rosenbrock(args):
#   x, y = args
#   a = 1
#   b = 100
#   return (a - x) ** 2 + b * (y - x ** 2) ** 2

def schwefel(X):
  # https://www.sfu.ca/~ssurjano/Code/schwefr.html
  d = len(X)
  sum = np.sum(X * np.sin(np.sqrt(np.abs(X.astype('float')))))
  y = 418.9829 * d - sum
  return y