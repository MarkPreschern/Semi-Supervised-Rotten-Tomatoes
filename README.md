# Semi-Supervised-Rotten-Tomatoes

**_Applies the Expectation Maximization algorithm on semi-supervised Rotten Tomatoes data, classifying sentences as either positive or negative reviews_**

*To alter performance, provide the following program arguments (optional):*
- '--semiSupervised (True/False)': Whether to consider the semi-supervised data in the file, or to do a completely unsupervised run
- '--fixedSeed (True/False)': Whether to perform the algorithm on a fixed seed for Random
- '--iterations (Positive Integer)': The number of iterations the EM algorithm will perform
- '--topWords (Positive Integer)': The number of words/bigrams that are printed per class
- '--naiveBayes (True/False)': Whether to use the Naive Bayes model or the Markov model of bigrams

*To run the program from the console, type 'python RottenTomatoesClassifier (optional program arguments)' and ensure that 'trainEMsemisup.txt is in the same directory as RottenTomatoesClassifier.py'*
