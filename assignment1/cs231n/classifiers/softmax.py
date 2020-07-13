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
    num_train, num_classes = scores.shape
    # numeric stability, for each row, shift each entry left by row max
    scores[range(num_train)] -= scores.max(axis=1, keepdims=True)
    scores = np.exp(scores)
    
    for image in range(num_train):
        s_i = scores[image]
        correct_class_score = s_i[y[image]]
        score_sum = s_i.sum()
        loss -= np.log(correct_class_score / score_sum)
        
        for class_ in range(num_classes):
            p = s_i[class_] / score_sum
            coeff = p - (class_ == y[image])
            dW[:, class_] += coeff * X[image,:]
            
    loss /= num_train
    loss += reg * (W**2).sum()
    
    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X@W
    num_train, num_classes = scores.shape
    scores[range(num_train)] -= scores.max(axis=1, keepdims=True)
    scores = np.exp(scores)
    
    correct_scores = scores[range(num_train), y]
    score_sums = scores.sum(axis=1)
    probs = correct_scores / score_sums
    log_probs = np.log(probs)
    
    loss = -log_probs.sum()
    loss /= num_train
    loss += reg * (W**2).sum()
    
    scores[range(num_train)] /= score_sums.reshape(-1,1)
    scores[range(num_train), y] -= 1
    dW = X.T @ scores
    dW /= num_train
    dW += 2 * reg * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
