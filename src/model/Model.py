from abc import ABC, abstractmethod


# Either a NaiveBayesModel or a MarkovModel
class Model(ABC):
    # Times (in expectation) that we need to see a word in a cluster
    # before we think it's meaningful enough to print in the summary
    MIN_TO_PRINT = 15.0
    # Probability of either a unigram or bigram that hasn't been seen
    # have make this generous since we're not using logs
    OUT_OF_VOCAB_PROB = 0.000001

    # the probability counts for each class
    classCounts = [0.0]*2
    # the total number of words per class is calculated as the probability of a word in the class
    # times the number of words in the given sentence
    totalWords = [0.0]*2
    # the probability of a given word in each class
    wordCounts = [{} for i in range(2)]

    @abstractmethod
    # Updates the model given a sentence and its probability of belonging to each class
    def update(self, sentence, probs):
        return

    @abstractmethod
    # Classifies a new sentence using the data in the model
    def classify(self, sentence):
        return

    @abstractmethod
    # printTopWords: Print n words/bigrams with the highest
    # Pr(thisClass | word/bigram) = scale Pr(word/bigram | thisClass)Pr(thisClass)
    # but skip those that have appeared (in expectation) less than
    # MIN_TO_PRINT times for this class (to avoid random weird words/bigrams
    # that only show up once in any sentence)
    def printTopWords(self, n):
        return

    # returns the probability of this class
    def classProbability(self, classIndex):
        total = 0
        for value in self.classCounts:
            total += value
        return self.classCounts[classIndex] / total

    # adjusts the current proportional probabilities to sum to 1
    @staticmethod
    def normalize(probs):
        pSum = 0
        for p in probs:
            pSum += p
        for i, p in enumerate(probs):
            probs[i] = probs[i] / pSum
        return probs
