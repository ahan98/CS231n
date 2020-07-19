## Data Preprocessing

- **Mean subtraction**
    - subtracting the mean centers the data around the origin
- **Normalization**
    - there are various ways to normalize data (see
        [Wiki](https://en.wikipedia.org/wiki/Feature_scaling) on feature
        scaling), but normalization makes most when the features of the data
        have different scales/units, and when each feature contributes to the
        algorithm's output equally (as in regression and distance-based
        algorithms)
    - from [AnalyticsVidhya](https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/)
        - "To ensure that the gradient descent moves smoothly towards the minima
            and that the steps for gradient descent are updated at the same rate
            for all the features, we scale the data before feeding it to the
            model."
        - tree-based algorithms are fairly *insensitive* to feature scaling
            because the decision at each node depends only on that particular
            node's feature
- **PCA & Whitening**
    - (come back to this section after learning some more linear algebra)

## Weight Initialization

