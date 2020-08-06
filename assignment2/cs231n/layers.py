from builtins import range
import numpy as np
from .im2col_alex import *

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N,D = x.shape[0], w.shape[0]
    x_reshape = np.reshape(x, (N,D))
    out = (x_reshape @ w) + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # backprop out = (x @ w) + b, wrt to w
    dx = dout @ w.T # (N,D)
    dx = np.reshape(dx, x.shape)

    # backprop out = (x @ w) + b, wrt to w
    x_reshape = np.reshape(x, (x.shape[0],-1))
    dw = x_reshape.T @ dout # (D,M)

    # backprop out = (x @ w) + b, wrt to b
    # note that b is broadcast to (N,M), i.e. re-express out as
    # out = (x @ w) + np.ones((N,1)) @ b
    db = np.sum(dout, axis=0) # equivalent to np.ones((1,N)) @ dout

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = x.copy()
    out[out < 0] = 0

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = (x > 0).astype('float') * dout

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # see: https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html

        mu = np.mean(x, axis=0)  # 1,D
        # equivalent to 1/N * np.sum(x, axis=0)

        var = np.var(x, axis=0)  # 1,D
        # equivalent to 1/N * np.sum(xmu**2, axis=0)

        sqrtvar = np.sqrt(var + eps) # 1,D

        ivar = 1/sqrtvar # 1,D

        xmu = x - mu # N,D

        xhat = xmu * ivar # N,D

        gammax = gamma * xhat # N,D

        out = gammax + beta # N,D

        cache = (xhat, gamma, xmu, ivar, sqrtvar, var, eps)

        running_mean = momentum * running_mean + (1-momentum) * mu
        running_var = momentum * running_var + (1-momentum) * var

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # NOTE: input is normalized using running mean/variance from TRAINING
        x_hat = (x - running_mean) / (np.sqrt(running_var + eps))
        out = (gamma * x_hat) + beta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, D = dout.shape
    (xhat, gamma, xmu, ivar, sqrtvar, var, eps) = cache

    # We sum over axis 0 so that the dimensions of dbeta match beta.
    # dbeta represents how much each individual element of beta affects the
    # output. In the simplest case, imagine xhat is a column vector, and dbeta
    # is a scalar. Then we have out = gamma * xhat + beta, and dout/dbeta is 1.
    #
    # Imagine the case where beta is a scalar, and out is a length-N column
    # vector. Then we want to find dout1/dbeta, ..., doutN/dbeta. We start
    # with beta, map it to a bunch of partials with respect to beta (i.e., each
    # douti/dbeta), and then map those back to a space with the same dimensions
    # as beta. That final mapping back to beta's dimensions is done by summing
    # up over over all intermediate products douti * dout/dbeta.
    #
    # The reason we sum over these per-dimension partials is due to the
    # Multivariate Chain Rule (see: https://youtu.be/hFvBZf-Jx28)

    # backprop out = gammax + beta
    dbeta = np.sum(dout, axis=0)
    dgammax = dout

    # backprop gammax = gamma * xhat
    dgamma = np.sum(xhat * dgammax, axis=0)
    dxhat = gamma * dgammax

    # backprop xhat = xmu * ivar
    dxmu = ivar * dxhat # ivar is broadcast to NxD
    divar = np.sum(xmu * dxhat, axis=0)

    # backprop ivar = 1/sqrtvar = (sqrtvar)^-1
    dsqrtvar = -(sqrtvar**-2) * divar

    # backprop sqrtvar = sqrt(var + eps) = (var + eps)^(0.5)
    dvar = 0.5 * (var + eps)**(-0.5) * dsqrtvar

    # backprop var = 1/N * np.sum(sq, axis=0)
    dsq = 1/N * dvar * np.ones((N,D))

    # backprop sq = xmu^2
    dxmu += 2 * xmu * dsq # accumulate gradient from multiple branches

    # backprop xmu = x - mu
    dx = dxmu
    dmu = -np.sum(dxmu, axis=0)

    # backprop mu = 1/N * np.sum(x, axis=0)
    dx += 1/N * np.ones((N,D)) * dmu

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    (xhat, gamma, xmu, ivar, sqrtvar, var, eps) = cache

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(xhat * dout, axis=0)

    N,D = dout.shape

    # NOTE: derivation in BatchNormalization.ipynb
    dxhat = dout * gamma
    dx1 = xhat * np.sum(dxhat * xhat, axis=0)
    dx2 = np.sum(dxhat, axis=0)
    dx = ivar * (dxhat - (dx1 + dx2)/N)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N,D = x.shape

    # compute PER-DATA (i.e. per-row) mean
    mu = np.mean(x, axis=1).reshape(N,-1)  # N,1
    # equivalent to 1/D * np.sum(x, axis=1).reshape(N,-1)

    var = np.var(x, axis=1).reshape(N,-1)  # N,1
    # equivalent to 1/D * np.sum(xmu**2, axis=1).reshape(N,-1)

    sqrtvar = np.sqrt(var + eps)  # N,1
    ivar = 1/sqrtvar  # N,1
    xmu = x - mu  # N,D
    xhat = xmu * ivar # N,D
    gammax = gamma * xhat # N,D
    out = gammax + beta # N,D
    cache = (xhat, gamma, xmu, ivar, sqrtvar, var, eps)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    (xhat, gamma, xmu, ivar, sqrtvar, var, eps) = cache

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(xhat * dout, axis=0)

    dxhat = dout * gamma
    N,D = dout.shape

    dx1 = xhat * np.sum(dxhat * xhat, axis=1).reshape(N,-1)
    dx2 = np.sum(dxhat, axis=1).reshape(N,-1)
    dx = ivar * (dxhat - (dx1 + dx2)/D)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mask = cache[1]
        # upstream gradient is zeroed out through cells which were masked to 0
        dx = dout * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.


    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    stride, pad = conv_param["stride"], conv_param["pad"]
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape

    x_pad = np.pad(x, ((0,0), (0,0), (pad, pad), (pad,pad)), "constant")

    assert (H - HH + 2*pad) % stride == 0
    assert (W - WW + 2*pad) % stride == 0
    H_prime = int(1 + (H - HH + 2*pad) / stride)
    W_prime = int(1 + (W - WW + 2*pad) / stride)
    out = np.zeros((N, F, H_prime, W_prime))

    for n in range(N):
        for f in range(F):
            for row in range(H_prime):
                x_i = stride * row        # starting row of local window in x
                for col in range(W_prime):
                    x_j = stride * col    # starting col of local window in x
                    for c in range(C):
                        x_view = x_pad[n, c, x_i : x_i+HH, x_j : x_j+WW]
                        out[n, f, row, col] += np.sum(w[f, c, :, :] * x_view)

                    out[n, f, row, col] += b[f]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # unpack params
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    N, F, H_prime, W_prime = dout.shape
    pad, stride = conv_param["pad"], conv_param["stride"]

    # re-pad x
    x_pad = np.pad(x, ((0,0), (0,0), (pad, pad), (pad,pad)), "constant")

    # init gradients
    dx_pad, dw, db = np.zeros_like(x_pad), np.zeros_like(w), np.zeros_like(b)

    for n in range(N):
        for f in range(F):
            db[f] += np.sum(dout[n, f, :, :])
            for out_i in range(H_prime):
                x_i = stride * out_i
                for out_j in range(W_prime):
                    x_j = stride * out_j
                    for c in range(C):
                        x_idxs = n, c, slice(x_i, x_i + HH), slice(x_j, x_j + WW)
                        w_idxs = f, c
                        cur_dout = dout[n, f, out_i, out_j]

                        dw[w_idxs] += x_pad[x_idxs] * cur_dout
                        dx_pad[x_idxs] += w[w_idxs] * cur_dout

    # lastly, backprop x <- x_pad
    dx = dx_pad[:, :, pad : pad+H, pad : pad+W]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    PH, PW = pool_param["pool_height"], pool_param["pool_width"]
    stride = pool_param["stride"]
    N, C, H, W = x.shape

    H_prime = int(1 + (H - PH) / stride)
    W_prime = int(1 + (W - PW) / stride)
    out = np.zeros((N, C, H_prime, W_prime))

    for n in range(N):
        for out_i in range(H_prime):
            x_i = stride * out_i        # starting row of local window in x
            for out_j in range(W_prime):
                x_j = stride * out_j    # starting col of local window in x
                for c in range(C):
                    x_view = x[n, c, x_i : x_i+PH, x_j : x_j+PW]
                    out[n, c, out_i, out_j] = x_view.max()


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, pool_param = cache
    PH, PW = pool_param["pool_height"], pool_param["pool_width"]
    stride = pool_param["stride"]
    N, C, H, W = x.shape
    N, C, H_prime, W_prime = dout.shape

    dx = np.zeros_like(x)

    for n in range(N):
        for out_i in range(H_prime):
            x_i = stride * out_i
            for out_j in range(W_prime):
                x_j = stride * out_j
                for c in range(C):
                    x_view = x[n, c, x_i : x_i+PH, x_j : x_j+PW]
                    max_i, max_j = np.unravel_index(np.argmax(x_view), x_view.shape)
                    # convert argmax for local window to argmax for entire x
                    max_i += x_i
                    max_j += x_j
                    dx[n, c, max_i, max_j] += dout[n, c, out_i, out_j]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Notice the shapes of gamma and beta are (C,). This suggests we treat C as
    # our new D, as in regular batchnorm, which takes in a batch of shape (N,D).
    # When we reshape x this way, our batch becomes N*H*W "images" of depth C.
    # We can imagine this as treating each pixel across all its channels as a
    # single batch sample. Now, mean and variance are C-dimensional. For
    # example, the mean of this pixel batch is a C-dimensional vector denoting
    # the average pixel in each channel.
    #
    # In the typical fully connected case of batchnorm, per-image normalization
    # makes sense because every image is multiplied by the same weights .
    # In CNNs however, images in the same channel are convolved with the same
    # weights, so the distributions are more likely to be consistent within the
    # same channel, but different across different channels. Therefore,
    # we have to normalize per-channel since the weights in each channel affect
    # the distribution of pixel values in each channel differently.
    #
    # Original batchnorm paper: https://arxiv.org/pdf/1502.03167.pdf
    # "For convolutional layers, we additionally want the normalization to obey
    # the convolutional property – so that different elements of the same
    # feature map, at different locations, are normalized in the same way. To
    # achieve this, we jointly normalize all the activations in a minibatch,
    # over all locations." (p.5)

    N, C, H, W = x.shape
    x_t = x.transpose((0, 2, 3, 1)).reshape(-1, C)  # (N, H, W, C) -> (N*H*W, C)
    out, cache = batchnorm_forward(x_t, gamma, beta, bn_param)
    out = out.reshape(N, H, W, C).transpose((0, 3, 1, 2))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = dout.shape
    dout = dout.transpose((0,2,3,1)).reshape(-1, C)
    dx, dgamma, dbeta = batchnorm_backward_alt(dout, cache)
    dx = dx.reshape(N, H, W, C).transpose((0, 3, 1, 2))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.  In contrast to
    layer normalization, group normalization splits each entry in the data into
    G contiguous pieces, which it then normalizes independently.  Per feature
    shifting and scaling are then applied to the data, in a manner identical to
    that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # In batchnorm, we normalize each image across all pixels (features).
    # In layernorm, we normalize each pixel across all images.
    # In groupnorm, we normalize each group across all images.
    #
    # Therefore, we want to reshape x into (A, B), where A is the number of
    # groups, and B is the number of pixels in each group. Then we perform a
    # "spatial layernorm", normalizing each group across all images, since each
    # group can be thought of as a "super channel/pixel/feature".
    #
    # Notice the notebook mentions that groupnorm assumes the channels/features
    # in each group contribute to the pixels in each group equally (unlike in
    # spatial batchnorm). This makes a lot of sense for some methods which
    # already exist in Computer Vision. For example, in HOG, "after computing
    # histograms per spatially local block, each per-block histogram is
    # normalized."
    #
    # Now groupnorm is just like layernorm, except we have to reshape the
    # normalized input back to 4 dimensions, so each group can be broadcasted
    # with gamma and beta.
    #
    # NOTE: keepdims=True keeps mean and var as a 2-D output. We could
    # choose not to use keepdims, but we'd have to reshape mean and var to
    # (A, 1) (i.e. transpose row vector into column vector), since x has shape
    # (A, B). This reshaping is to allow broadcasting with gamma and beta.

    N, C, H, W = x.shape
    x_group = x.reshape((N * G, C//G * H * W))

    mu, var = x_group.mean(axis=1, keepdims=True), x_group.var(axis=1, keepdims=True)
    xmu = x_group - mu
    sqrtvar = np.sqrt(var + eps)
    ivar = 1 / sqrtvar
    xhat = xmu * ivar

    xhat.shape = (N, C, H, W)
    gamma.shape = beta.shape = (1, C, 1, 1)
    out = xhat * gamma + beta

    cache = (xhat, gamma, xmu, ivar, sqrtvar, var, eps, G)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # In groupnorm forward pass, we reshape into 2d, normalize, then reshape
    # back to 4d, and finally compute out using gamma and beta.
    #
    # So in the backward pass, we want to compute dbeta and dgamma while
    # everything is still 4d, then reshape to 2d, compute dx, and finally
    # reshape back to 4d.
    #
    # Computing dx is literally the exact same as in layernorm backwards pass,
    # especially since we reshape back to 2d first. Computing dbeta and dgamma
    # is still almost the exact same, since we are summing over every axis that
    # is NOT the channel axis (so axes 0, 2, and 3). We sum over all the
    # non-channel axes so that the shape of dbeta matches beta, and the
    # intuitive interpretation of this sum is the same as in backprop of normal
    # batchnorm and layernorm.

    (xhat, gamma, xmu, ivar, sqrtvar, var, eps, G) = cache
    N, C, H, W = dout.shape

    dbeta = np.sum(dout, axis=(0,2,3), keepdims=True)
    dgamma = np.sum(xhat * dout, axis=(0,2,3), keepdims=True)

    dxhat = dout * gamma
    dxhat.shape = xhat.shape = (N * G, C//G * H * W)

    dx1 = xhat * np.sum(dxhat * xhat, axis=1, keepdims=True)
    dx2 = np.sum(dxhat, axis=1, keepdims=True)
    dx = ivar * (dxhat - (dx1 + dx2) / (C//G * H * W))
    dx.shape = (N, C, H, W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
