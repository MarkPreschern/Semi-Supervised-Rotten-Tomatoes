# Semi-Supervised-Rotten-Tomatoes

### Applies the Expectation Maximization algorithm on semi-supervised Rotten Tomatoes data, classifying sentences as either positive or negative reviews

To alter performance, change the following static variables:
- SEMISUPERVISED: Whether to consider the semi-supervised data in the file, or to do a completely unsupervised run
- FIXED_SEED: Whether to perform the algorithm on a fixed seed for Random
- ITERATIONS: The number of iterations the EM algorithm will perform
- USE_NAIVE_BAYES: Whether to use the Naive Bayes model or the Markov model of bigrams
