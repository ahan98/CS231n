# Neural Networks Part 1

*These notes accompany the original course notes
[here](https://cs231n.github.io/neural-networks-1/).*

## A Simple Example

Before fully defining the term "neural network," we introduce a simple example,
separate from the biological analogy of a neuron.

An example of a simple neural network is:

$$s = W_2 \text{max}(0, W_1 x).$$

In words, this neural network multiples the input $x$, say a column vector
corresponding to a single input object of shape [1 x $D$], with learnable weight
matrix $W_1$, which could be of shape [$C$ x $D$], where $C$ is the number of
classes, and $D$ is the dimension of the data.

Notice the importance of the non-linear $\text{max}$ function. If this function
were not present in the calucation of the class score $s$, then the computation
would simply be a linear product of the weight matrices and input.

Lastly, the gradients of $W_i$ are learned with backpropagation, and those
gradients are used to update and optimize $W_i$ via SGD.

<ins>Note:</ins> A 3-layer neural network could be expressed as $s = W_3
\text{max}(0, W_2 max(0, W_1 x),$ so we can begin to see how a neural network of
many layers is just a consecutive chaining of a non-linear activation function
applied to a matrix multiplication, where the results of each matrix
multiplication is stored in intermediate neurons, i.e. hidden layers.

## Modeling a Single Neuron

Below is the code taken directly from the original notes, which models the
forward pass of a single neuron. The first line, beginning with `cell_body_sum`
multiplies the weight matrix $W_i$ with input $x$, and the resulting value (in
this case, a single real value) is passed into the non-linear activation
function, from which we get the final output.

This forward pass is analogous to the biological process of a neuron
firing a signal. Roughly speaking, a signal is transmitted *into* a neuron via
an axon. That signal interacts multiplicatively with the strength at a synapse
(corresponds to $w_i x_i$). All these multiplicative interactions are then
summed up in the neuron's cell body, completing the matrix multiplication.

```python
class Neuron(object):
  def forward(self, inputs):
    """ assume inputs and weights are 1-D numpy arrays and bias is a number """
    cell_body_sum = np.sum(inputs * self.weights) + self.bias
    firing_rate = 1.0 / (1.0 + math.exp(-cell_body_sum)) # sigmoid activation function
    return firing_rate
```

### Single Neuron as a Linear Classifier

The computation of the modeled neuron can be used to simulate linear
classification, by applying the appropriate loss/activation function to the
neuron's output. This is because the new result then tells us to which class the
input belongs, hence simulating linear classification.

### Common Activation Functions

- Sigmoid
    - (+) bounds values between [0,1], which intuitive maps to a neuron being
        "dead" (0), or firing at maximum frequency (1)
    - (-) saturated at the lower/upper tails, which causes local gradients in
        those regions to be (near-zero)
        - with a very small local gradient, during backpropagation, each
            recursive local gradient will be zeroed out
        - extra care is needed when initializing the weights to prevent
            saturation
    - (-) function is non zero-centered, which is bad because if the data coming
        into a neuron is monotone (all positive, or all negative) then the
        signs of the weight gradients will alternate from layer-to-layer and
        maintain monotonicity
        - however, note that over an entire batch of data, after summing across
            all the gradient updates, the resulting weights are not necessarily
            monotone, since the magnitude of each cellwise update can differ

- Tanh
    - this function is just a scaled version of sigmoid, with values bounded
        between [-1,1] instead of [0,1]
    - this function still suffers from saturation, but the data is now
        zero-centered

- ReLU - Rectified Linear Unit
    - this is the same non-linear function used in SVM loss
    - (+) faster computation compared to the previous functions, leading to
        faster convergence of SGD
    - (-) local gradients thresholded to zero can become "dead spots" during
        training
        - for example, a learning rate that's too high can irreversibly update
            the many of the weights such that particular regions will remain 0
            for the rest of training

- Leaky ReLU
    - this function attempts to mitigate the potential dead spots of ReLU by
        multiplying the value by a very small negative constant $\alpha$ when
        the value is negative, rather than thresholding it immediately to 0

- Maxout
    - a generalization of ReLU and leaky ReLU
    - retains the pros of ReLU, withtout the drawbacks of dead neurons
    - however, a new drawback is introduced in the increased complexity of
        parameters (doubled)

## Neural Network Architectures

### Layer-wise Organization

**Fully-connected:** Given layer 1 with $n$ neurons, and layer 2 with $m$
neurons, then there are $n * m$ directed edges from each neuron in layer 1 to
each neuron in layer 2.

**Naming conventions:** An N-layer neural network consists of 1 input layer, 1
output layer, with N-1 hidden layers in between.
- SVM and softmax are special cases of single-layer NNs, since the input is
    mapped directly to the output layer

**Output layer:** This layer typically does not have an activation function, so
the values stored in this layer correspond to integer indices (in
classification) or real values (in regression).

**Size:** The size of a NN is the total number of non-input neurons, plus the
number of edges.

### Representational Power
Viewing an entire neural network as a family of functions, each parameterized by
the local weights, we ask: "What families of functions *can't* be modeled by a
NN?"

It turns out that any NN with at least 2 layers (i.e., at least 1 hidden layer)
can approximate any continuous function with arbitrary precision. I found
[this](https://www.youtube.com/watch?v=Ijqkc7OLenI) YouTube video by Michael
Nielsen excellent for explaining the intuition behind how a 2-layer NN could be
used as a universal approximator.

However, this fact alone is not very useful in practice because if use, for
example, the function $g(x) = \sum_i c_i \mathbb{1}(a_i < x < b_i)$, for
parameter vectors $a,b,c$, such a function doesn't provide much information in
the way of classifying $x$ (the $\mathbb{1}$ is supposed to be the indicator
function, which I couldn't get to work with MathJax, so that may appear as just
a normal 1).

But if we use existing statistical functions which are used to express data in
meaningful ways, then we can see that the ability to use neural networks as
universal approximators is useful in compactly expressing those functions which
are continuous.

Regarding "power," it is actually empirically observated that neural networks
with more layers work better than those with fewer. In practice, 3-layer NNs
work better than 2-layer NNs, but beyond 3 layers, the returns typically
diminish significantly. This is NOT the case with CNNs, which, intuitively, can
be explained as a result of images being composed hierarchically (e.g. "faces
are made up of eyes, which are made up of edges), so with additional hidden
layers storing information of these components, performance improves.

### Setting Number of Layers and Their Sizes
