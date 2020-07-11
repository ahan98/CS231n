from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X@W # scores[i][j] = s means image x_i has score s for class j
    N, C = scores.shape
    for image in range(N):
        scores[image] -= np.max(scores[image]) # numeric stability
        scores[image] = np.exp(scores[image])
        correct_class_score = scores[image][y[image]]
        loss -= np.log(correct_class_score / scores[image].sum())

    loss += reg * (W**2).sum()
    loss /= N

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # NOTE: loss formula is expressed differently from naive
    # loss_i = -f_(y[i]) + log(sum(e^(f_j)))
    # f_j denotes the score of the j-th class for x_i
    # y[i] denotes index of true class of x_i

    scores = X@W
    N,_ = scores.shape
    scores -= scores.max(axis=1, keepdims=True) # numerical stability
    # correct_scores[i] = -s means the score of the true class of x_i is s.
    # range iterates through each row. at row i, we take the y[i]-th element.
    correct_scores_neg = -scores[range(N), y]
    scores = np.exp(scores)

    correct_scores_neg += np.log(scores.sum(axis=1))
    loss = correct_scores_neg.sum()/N + reg * (W**2).sum()

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
