# Neural Networks Part 2
*Original course notes
[here](https://cs231n.github.io/neural-networks-2/)*

**TODO** (topics I don't yet understand at a basic level)
- PCA & Whitening

## Data Preprocessing

**Mean subtraction**
- subtracting the mean centers the data around the origin

**Normalization**
- there are various ways to normalize data (see
  [Wiki](https://en.wikipedia.org/wiki/Feature_scaling) on feature scaling), but
  normalization makes most sense when the features of the data have different
  scales/units, and when each feature contributes to the algorithm's output
  equally (as in regression and distance-based algorithms)
- from [AnalyticsVidhya](https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/)
    - "To ensure that the gradient descent moves smoothly towards the minima and
        and that the steps for gradient descent are updated at the same rate for
        all the features, we scale the data before feeding it to the model."
    - tree-based algorithms are fairly *insensitive* to feature scaling
        because the decision at each node depends only on that particular node's
        feature

**PCA & Whitening**
- TODO

## Weight Initialization
**Initialize all weights to zero** (*BAD!*)
- if all weights are zero, and all neurons output the same value, then the
    gradient of the weights with respect to each neuron will be the same, so
    each weight will receive the same parameter update

**Small random numbers**
- initialize each weight to be a small random value based on the size of the
    matrix
- `W = 0.01 * np.random.randn(D, H)`
- note that smaller initial weights are not necessarily better; for example,
    during backpropagation, small weights could minimize the local gradients too
    much, diminishing the information returned by that path

**Dividing the variances by 1/sqrt(n)**
- `W = np.random.randn(n) / sqrt(n)`
- `n` is the "fan-in," i.e. the number of inputs (from the previous layer)
    into a neuron
- this improves upon the small random number heuristic because now, the weights
    are not proportional to the size of a matrix (where a large matrix would
    have larger weights on average)
- see Stanford's notes for derivation
- note in practice, dividing by `sqrt(2.0/n)` works better

**Sparse Initialization**
- initalize all weights to zero, but create asymmetry by randomizing the number
    of each neuron's connections
- in practice, this would be done by taking a zero matrix, and then in each row,
    sample a subset of weights, and assign to the selected weights values drawn
    from a (small) gaussian distribution

**Initalizating the biases**
- In practice, initializating all biases to zero is fine because asymmetry is
    already provided to the weights

**Batch Normalization**
- we know that normalizing the input layer can reduce training time by allowing
    the gradient to converge more quickly
- batch normalization attempts to apply this benefit to the hidden layers of the
    NN
    - the intuition is that normalization is a convex, differentiable function,
        leading to faster convergence because the local optimum is the global
        optimum
    - detailed paper by [Ioffe and Szegedy](https://arxiv.org/abs/1502.03167)
- roughly speaking, the PRE-activation hidden layer values (so for example, the
    values of a hidden layer prior to applying ReLU) are normalized (subtract
    mean, then divide by standard deviation), and then further shifted/scaled by
    two *learnable* parameters gamma and beta
    - gamma and beta allow the distributions room to change from layer-to-layer,
        since this property might be desirable depending on the problem
    - note that the changing of activation distribution between consecutive
        layers is known as "internal covariate shift"
- Andrew Ng's explanation [video](https://youtu.be/tNIpEZLv_eg)
    - Ng's follow-up [video](https://youtu.be/nUUqwaxLnWs) explaining *why*
        batch normalization is effective

## Regularization

Intuitively, regularization can be thought of as penalizing the complexity of
the model to prevent overfitting. For example, using L2 regularization
encourages the weights of the model to contribute more uniformly to the output,
as opposed to a only a few contributing and by a lot.

**L2**
- for every weight `w`, add `0.5 * lambda * w**2` to the regularization loss
    - the factor of `0.5` is just so that, in the gradient, the regularization
        term simplifies to `lambda * w`
- Stanford's explanation for the intuition behind why L2 regularization works
    well:
    - "The L2 regularization has the intuitive interpretation of heavily
        penalizing peaky weight vectors and preferring diffuse weight vectors.
        As we discussed in the Linear Classification section, due to
        multiplicative interactions between weights and inputs this has the
        appealing property of encouraging the network to use all of its inputs a
        little rather than some of its inputs a lot."
    - example: `x = [1,1,1,1]`, `w1 = [1,0,0,0]`, `w2 = [0.25, 0.25, 0.25,
        0.25]`
    - then `w1.T @ x == w2.T @ x == 1`, but the L2 penalty of `w2` is 1 and 0.25
        for `w1`, demonstrating the preference for diffuse weights explained
        above

**L1**
- an interesting property of L1 regularization is that the resulting weight
    matrix becomes very sparse, meaning each neuron uses only its most important
    features
    - consequently, this type of regularization makes the model resistant to
        noise in the data
- if you are not concerned about using specific features, L2, with its tendency
    to diffuse weights, generally outperforms L1

**Max norm constraints**
- after updating the weights normally, clamp the weights such that their L2 norm
    is bounded by some constant c, which is typically on the order of 3 or 4
    (i.e. 1e3, 1e4)
- by bounding the weights, how much the loss can increase after each update
    becomes limited as well

**Dropout**
- in the forward pass, as we move compute the neurons for the current layer,
    each neuron has a probability `p` of being dropped
- a "dropped" neuron is zeroed out
- this can be interpreted as sampling a subset of neurons from a network
- note that during prediction, the computation of each layer must be scaled by
    `p` to achieve the same expected outcome for each neuron
    - for example, if one neuron outputs value `x`, then it's expected output is
        `px + (1-p)0 = px`
- to make prediction performance better, this scaling can be applied during the
    forward pass of training, via **inverse dropout**, where each of the layers
    are scaled by `1/p`

**Noise in the forward pass**
- the scaling by `p` during dropout *analytically* marginalizes noise
    (analytical because we can immediately compute the expected activation for
    each neuron)
- we can also *numerically* marginalize noise by taking multiple random samples
    of neurons from a network, and then average their outputs
- in this context, to "marginalize" noise means to reduce the effects of noisy
    input on the output, thus preventing overfitting

**Bias regularization**
- unlike the weights, the bias does not interact with the data multiplicatively,
    so they do not control the influence of a data dimension on the output
- therefore, regularizing bias is uncommon
- however, it rarely worsens performance significantly, likely due to the fact
    that there are much fewer bias terms than there are weights, so the model
    shoul

**Per-layer regularization**
- it is uncommon to apply different regularization methods to different layers,
    and there is not yet much research on this technique

**Best practice**
- use global L2 regularization with cross-validation (to tune `lambda`)
- use dropout after each layer as well, with `p = 0.5` being a good default,
    which can be tuned as a hyperparameter

## Loss Functions

**Large number of classes**
A difficulty we encounter with some data loss functions such as softmax is that
the function can be expensive to compute if there is a large number of classes.
In some domains, such as NLP, labels are arranged into a tree structure, where
each path corresponds to a label, and softmax actually trains at the individual
node level, effectively treating the left and right branches as the possible
"classes."

**Attribute Classification**
If our problem demands that one object may have multiple labels (such as an
Instagram photo having multiple hashtags out of a larger subset of hashtags), we
can use a binary classifier to iterate over the set of possible attributes, and
determine one at a time whether the i-th data object contains that attribute or
not.  See the Stanford [notes](https://cs231n.github.io/neural-networks-2/) for
the loss function.

Another solution we could use is to train a logistic regression classifier for
each attribute. We say an object possesses an attribute if its conditional
probability of having that attribute is greater than 0.5, and does not,
otherwise. Now, we have two possible labels to denote whether the i-th object
has attribute j (`y_ij > 0` if object i has attribute j, and `y_ij < 0`
otherwise.

A nice property of this binary regression method, which uses the sigmoid
function, is that the derivative is simple and intuitive.

**Regression** In regression problems, we want to compute a real-valued output
for each input.  L2 norm is commonly used as loss, where we measure the distance
between the predicted and ground truth values. In practice, we actually square
the norm (so it just becomes a linear combination of squared terms), which
simplifies the gradient. And, we can do so without needing to adjust any of the
parameters, since squaring is monotone.

See this Kaggle
[article](https://www.kaggle.com/residentmario/l1-norms-versus-l2-norms), which
provides a concise comparison of L1 and L2 norms.

Note that the gradient of the i-th loss with respect to the j-th dimension for
L2 norm is easily derived from the linear combination of squares, and for L1
norm, it is +/- 1 if f_j is positive or negative (where f_j is the predicted
regression value). In other words, for L2, the gradient is directly proportional
to the difference between real and predicted, or it is just the sign of the
output.

<ins>Note:</ins> Regularization for regression is much harder to optimize than
something like softmax for classification, where the exact meaning of the scores
is not so important, compared to the relative differences between an
input's scores. That is, because regression requires a single real value, that
value has to be accurately computed, taking into account any augmentations (like
scaling by `p` in dropout).
- therefore, before framing the problem as regression, make sure it's actually
    necessary
- instead of regression, we could, for example, reframe the problem as
    classification by compartmentalizing the output into bins (such as using a
    5-star system instead of a floating-point rating)
- if we can reframe the problem into classification, we get the added benefit of
    seeing the distribution of scores across the classes, as well as the
    confidence of the predicted class (like in softmax), and not just a single
    value with no associated confidence

**Structured Prediction**
This is a kind of classification problem where the classes themselves can be
complex structures (e.g., trees, graphs, etc.). Typically, the space of possible
structures is not easily enumerable. This problem is usually solved using SVM,
where the delta margin is denotes the difference between the correct structure
and the "most incorrect" predicted structure. Unlike gradient descent, which is
an unconstrained optimization problem ("unconstrained" in the sense that the
gradients can be any real value), structured prediction, roughly speaking, uses
solvers which take advantage of the assumptions which simplify the space of
structures for the specific problem.
