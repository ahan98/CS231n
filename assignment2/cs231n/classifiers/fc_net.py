from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.params["W1"] = np.random.normal(scale=weight_scale, size=(input_dim, hidden_dim))
        self.params["b1"] = np.zeros((hidden_dim,))
        self.params["W2"] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
        self.params["b2"] = np.zeros((num_classes,))

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out1, cache1 = affine_forward(X, self.params["W1"], self.params["b1"])
        # out1 = output of first hidden layer
        # cache1 = X, W1, b1
        out1[out1 < 0] = 0
        scores, cache2 = affine_forward(out1, self.params["W2"], self.params["b2"])
        # cache2 = out1, W2, b2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(self.params["W1"]**2) + np.sum(self.params["W2"]**2))

        grads = {}

        # backprop scores = out1 @ W2 + b2
        dout1, grads["W2"], grads["b2"] = affine_backward(dscores, cache2)
        grads["W2"] += self.reg * self.params["W2"]
        dout1 *= (out1 > 0) # backprop relu on out1

        # backprop h = X @ W1 + b1
        _, grads["W1"], grads["b1"] = affine_backward(dout1, cache1)
        grads["W1"] += self.reg * self.params["W1"]

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.dtype = dtype
        self.params = {}

        # params added by me
        self.num_layers = len(hidden_dims) + 1 # hidden + output

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # store params for hidden layers
        last_input_dim = input_dim
        for i, h in enumerate(hidden_dims):
            idx = str(i + 1)
            self.params["W" + idx] = np.random.normal(scale=weight_scale,\
                                                        size=(last_input_dim, h))
            self.params["b" + idx] = np.zeros((h,))

            if self.normalization:
                self.params["gamma" + idx] = np.ones((h,))
                self.params["beta" + idx] = np.zeros((h,))

            last_input_dim = h

        # store params for output layer
        idx = str(self.num_layers)
        self.params["W" + idx] = np.random.normal(scale=weight_scale,\
                                                    size=(last_input_dim, num_classes))
        self.params["b" + idx] = np.zeros((num_classes,))

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        outputs, fc_cache, norm_cache, relu_cache = self.sandwich_layers_forward(X)
        scores = outputs[-1]

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dscores = softmax_loss(scores, y)
        for i in range(1, self.num_layers + 1):
            idx = str(i)
            W = self.params["W" + idx]
            loss += 0.5 * self.reg * np.sum(W**2)
        # print(loss)

        grads = self.sandwich_layers_backward(dscores, fc_cache, norm_cache, relu_cache)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def sandwich_layers_forward(self, X):
        """
        Computes and saves the intermediate layers in the forward pass in the
        following order: X -> affine -> batch/layernorm -> ReLU -> dropout

        NOTE: Normalization and dropout are optional and apply only to hidden
        layers.

        The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
        examples, where each example x[i] has shape (d_1, ..., d_k). We will
        reshape each input into a vector of dimension D = d_1 * ... * d_k, and
        then transform it to an output vector of dimension M.

        Inputs:
        - X: A numpy array containing input data, of shape (N, d_1, ..., d_k)

        Outputs:
        - outputs: outputs for each layer (0,...,N)
        - fc_cache: inputs for each layer (0,...,N)
        """

        N = X.shape[0]
        X = X.reshape((N,-1))

        # for each of these caches, i = 0,1,...,N denotes layer index
        # 0 = input layer, N = output layer, 1..N-1 = hidden layers

        outputs = [None] * (self.num_layers + 1)
        outputs[0] = X

        # fc_cache[0] is None, since no inputs for input layer
        # fc_cache[j] = (outputs[j-1], W_j, b_j) for j = 1,...,N
        fc_cache = [None] * (self.num_layers + 1)

        # norm_cache[i] stores batch/layernormed output from outputs[i].
        # bn is not applied to input nor output layers, so norm_cache[0] and
        # norm_cache[-1] are both None.
        norm_cache = [None] * (self.num_layers + 1)

        # relu_cache[i] stores relu'd output from outputs[i] or norm_cache[i].
        # similar to norm_cache, relu_cache[0] = relu_cache[-1] = None.
        relu_cache = [None] * (self.num_layers + 1)

        for i in range(1, self.num_layers + 1):
            idx = str(i)
            W, b = self.params["W" + idx], self.params["b" + idx]
            out, fc = affine_forward(outputs[i-1], W, b)

            # transform hidden layers layers (i = 1,...,N-1)
            norm = relu = None
            if i < self.num_layers:

                if self.normalization:
                    gamma = self.params["gamma" + idx]
                    beta = self.params["beta" + idx]
                    bn_param = self.bn_params[i-1]

                    if self.normalization == "batchnorm":
                        out, norm = batchnorm_forward(out, gamma, beta, bn_param)
                    elif self.normalization == "layernorm":
                        out, norm = layernorm_forward(out, gamma, beta, bn_param)

                out, relu = relu_forward(out)

            outputs[i] = out
            fc_cache[i] = fc
            norm_cache[i] = norm
            relu_cache[i] = relu
            # dropout_cache[i] = drop

        return outputs, fc_cache, norm_cache, relu_cache#, dropout_cache

    def sandwich_layers_backward(self, dscores, fc_cache, norm_cache, relu_cache):
        grads = {}

        # We begin with the current layer as the output layer, for which there
        # is no batchnorm, ReLU, or dropout transformations.
        i = self.num_layers
        W, b = "W" + str(i), "b" + str(i)
        # compute gradients for INPUTS to current layer
        d_pre, grads[W], grads[b] = affine_backward(dscores, fc_cache[i])
        grads[W] += self.reg * self.params[W]

        # In the forward pass, the sequence of transformations on input X is:
        # X -> affine -> batch/layernorm -> ReLU -> dropout
        # So in the backward pass, we just undo each transformation (if
        # applicable) in the opposite order.
        for i in range(self.num_layers - 1, 0, -1):
            idx = str(i)
            W, b = "W" + idx, "b" + idx
            gamma, beta = "gamma" + idx, "beta" + idx

            # before, d_pre is the gradient for the input of layer i+1
            d_pre = relu_backward(d_pre, relu_cache[i])
            if self.normalization == "batchnorm":
                d_pre, grads[gamma], grads[beta] = batchnorm_backward_alt(d_pre, norm_cache[i])
            elif self.normalization == "layernorm":
                d_pre, grads[gamma], grads[beta] = layernorm_backward(d_pre, norm_cache[i])

            # if dropout_cache[i]:
            #     d_pre = dropout_backward(d_pre, dropout_cache[i])
            d_pre, grads[W], grads[b] = affine_backward(d_pre, fc_cache[i])
            # after, d_pre is the gradient for the input of layer i

            grads[W] += self.reg * self.params[W]

        return grads
