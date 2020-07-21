from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_train, num_dims = X.shape
    num_classes = W.shape[1] # C
    loss = 0.0

    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]

        for j in range(num_classes):
            if j == y[i]:
                continue

            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:,j] += X[i,:]
                dW[:,y[i]] -= X[i,:]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # As suggested by hint, calculation of gradient is done in the for-loop
    # structure used to compute loss.

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    delta = 1.0
    scores = X @ W
    num_train  = X.shape[0]
    scores -= scores[range(num_train), y].reshape(-1,1)
    scores[scores != 0] += delta

    # for each scores[i][j] > 0, add X[i,:] to dW[:,j]
    loss = scores[scores > 0].sum() / num_train
    loss += reg * (W**2).sum()

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # write half-vectorized first to develop intuition for full-vectorized
    # for image in range(N):
    #     mask = np.where(scores[image] > 0)[0]
    #     image_to_col = X[image,:].reshape(D,1)
    #     dW[:,mask] += image_to_col
    #     dW[:,y[image]] -= len(mask) * X[image]

    coeffs = np.greater(scores, 0).astype('int')
    coeffs[range(num_train), y] -= coeffs.sum(axis=1)
    # suppose coeffs[i][c] = k
    # if c is the true class of x_i, then this means image x_i had a positive
    # score for |k| classes.
    # else if k == 1, then x_i had a positive score with class c.
    # else if k == 0, then x_i had a score of 0 or less.
    # hence, coeffs[i][c] stores the coefficient of x_i in the formula for
    # the gradient of loss_i w.r.t W_c

    dW = X.T @ coeffs
    # the sum of the pairwise products of row d (in X.T) and col c (in coeffs)
    # will be stored in dW[d][c], which makes sense because this means
    # the d-th dimension has been added to class c count times, where count
    # is the sum of the coefficients in column c.

    # intuitively, this dot product simulates adding (or subtracting) image x_i
    # (as a column # vector) to each class for which x_i had a positive score.

    # We can imagine dW being filled in row-by-row, rather than column-by-column,
    # which is more intuitive and is what we did in the naive and half-
    # vectorized versions. When adding column-wise, we are adding whole images
    # to a class. But when adding row-wise, we are adding a specific dimension
    # from each image that scored positively with the current class.

    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
