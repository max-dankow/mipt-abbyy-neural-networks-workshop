import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  num_classes = W.shape[1]
  num_train = X.shape[0]
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  s = np.dot(X, W)

  for i in range(num_train):
    row = s[i] - np.max(s)
    sum_ = np.sum(np.exp(row))
    loss += -np.log(np.exp(row[y[i]]) / sum_)
    for k in range(num_classes):
      p = np.exp(row[k]) / sum_
      dW[:, k] += (p - (k == y[i])) * X[i]

  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * 2 * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  num_train = X.shape[0]
  dW = np.zeros_like(W)
  s = np.dot(X, W)
  s = s - np.max(s, axis=1, keepdims=True)
  exp = np.exp(s)
  exp_sum = np.sum(exp, axis=1, keepdims=True)
  p = exp / exp_sum
  loss = np.mean(-np.log(p[np.arange(num_train), y]))

  t = np.zeros_like(p)
  t[np.arange(num_train), y] = 1
  dW = X.T.dot(p - t)
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * 2 * W

  return loss, dW

