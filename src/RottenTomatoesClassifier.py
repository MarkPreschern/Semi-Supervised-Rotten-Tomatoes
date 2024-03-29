# Semi-supervised Rotten Tomatoes:
# Expectation-Maximization using Naive Bayes and Markov Models to do sentiment analysis.

# Input from train.tsv.zip at https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews
# gathered from Rotten Tomatoes.
#
# Format is PhraseID[unused]   SentenceID  Sentence[tokenized]
#
# Just a few sentiment are provided as this is semisupervised.
#
# We'll only use the first line for each SentenceID, since the others are
# micro-analyzed phrases that would just mess up our counts.
#
# After training, we'll identify the top words for each cluster by
# Pr(cluster | word) - the words that are much more likely in the cluster
# than in the general population - and categorize the new utterances.

import argparse
import random
import Lemmatizer
import Sentence
from model import NaiveBayes
from model import Markov


class RottenTomatoesClassifier:
    # The number of CLASSES to train the data to
    CLASSES = 2

    # whether to consider the semi-supervised data in the file, or to do a completely unsupervised run
    semi_supervised = True
    # whether to lemmatize the input sentences
    lemmatize = True
    # whether to perform the algorithm on a fixed seed for Random
    fixed_seed = False
    # the number of iterations the Expectation-Maximization algorithm will perform
    iterations = 200
    # the number of words/bigrams that are printed per class
    top_words = 10
    # whether to use the Naive Bayes model or the Markov model of bigrams
    naive_bayes = True
    # the current model being used
    model = NaiveBayes.NaiveBayes()

    # runs necessary steps to classify the rotten tomatoes data
    def run(self):
        self.parseArgs()
        sentences = self.getTrainingData()
        lemmatizedSentences = Lemmatizer.Lemmatizer().lemmatize(self.lemmatize, sentences)
        self.trainModels(lemmatizedSentences)
        self.model.printTopWords(self.top_words)
        self.classifySentences()

    # parses command line arguments
    def parseArgs(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-s", "--semiSupervised", type=self.strToBool, default=True)
        parser.add_argument("-l", "--lemmatize", type=self.strToBool, default=True)
        parser.add_argument("-f", "--fixedSeed", type=self.strToBool, default=False)
        parser.add_argument("-i", "--iterations", type=int, default=200)
        parser.add_argument("-t", "--topWords", type=int, default=10)
        parser.add_argument("-n", "--naiveBayes", type=self.strToBool, default=True)

        args = parser.parse_args()
        if args.semiSupervised is not None:
            self.semi_supervised = args.semiSupervised
        if args.lemmatize is not None:
            self.lemmatize = args.lemmatize
        if args.fixedSeed is not None:
            self.fixed_seed = args.fixedSeed
        if args.iterations is not None and args.iterations > 0:
            self.iterations = args.iterations
        if args.topWords is not None and args.topWords > 0:
            self.top_words = args.topWords
        if args.naiveBayes is not None:
            self.naive_bayes = args.naiveBayes
            self.model = NaiveBayes.NaiveBayes() if args.naiveBayes else Markov.Markov()

    # custom boolean operator type for argparse
    @staticmethod
    def strToBool(v):
        if isinstance(v, bool):
            return v
        elif isinstance(v, str):
            if v.lower() in ('yes', 'true', 't', 'y', '1'):
                return True
            elif v.lower() in ('no', 'false', 'f', 'n', '0'):
                return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # parses the training data
    @staticmethod
    def getTrainingData():
        with open("trainEMsemisup.txt") as f:
            sentences = []
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    if line[:3] == "---":
                        return sentences
                    else:
                        sentences.append(line)
        f.close()
        return sentences

    # applies the Expectation-Maximization algorithm on the chosen model
    def trainModels(self, sentences):
        # We'll start by assigning the sentences to random CLASSES.
        # 1.0 for the random class, 0.0 for everything else
        print("Initializing models....")
        naiveCLASSES = self.randomInit(sentences)
        # Initialize the parameters by training as if init were
        # the ground truth (essentially starting with M step)
        for key, value in naiveCLASSES.items():
            self.model.update(key, value)

        # for a set number of iterations, performs the expectation and maximization steps to update the model
        i = 0
        while i < self.iterations:
            print("EM round " + str(i))
            classes = {}
            # expectation step
            for sentence in sentences:
                classes[sentence] = self.model.classify(sentence)
            # maximization step
            self.model = NaiveBayes.NaiveBayes() if self.naive_bayes else Markov.Markov()
            for key, value in classes.items():
                self.model.update(key, value)
            i += 1

    #  randomly initializes the unsupervised data based on semi_supervised, CLASSES, and rng
    def randomInit(self, sentences):
        rand = random.seed(2019) if self.fixed_seed else random
        counts = {}
        for sentence in sentences:
            probs = []
            if self.semi_supervised and sentence.words[:2] == ":)":
                # class 1 = positive
                probs.append(0.0)
                probs.append(1.0)
                sentence = Sentence.Sentence(sentence.words[3:], sentence.lemmas[4:])
            elif self.semi_supervised and sentence.words[:2] == ":(":
                # class 0 = negative
                probs.append(1.0)
                probs.append(0.0)
                sentence = Sentence.Sentence(sentence.words[3:], sentence.lemmas[4:])
            else:
                baseline = 1.0 / self.CLASSES
                # slight deviation to break symmetry
                randomBumpedClass = rand.randrange(0, self.CLASSES)
                bump = 1.0 / self.CLASSES * 0.25
                if self.semi_supervised:
                    bump = 0.0
                i = 0
                while i < self.CLASSES:
                    if (i == randomBumpedClass):
                        probs.append(baseline + bump)
                    else:
                        probs.append(baseline - bump / (self.CLASSES - 1))
                    i += 1
            counts[sentence] = probs
        return counts

    # classifies the sentences
    def classifySentences(self):
        print("Classifying test sentences")
        onTestingData = False
        with open("trainEMsemisup.txt") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.strip()
                if line:
                    if onTestingData:
                        if line == "Negative:" or line == "Positive:":
                            print(line)
                        else:
                            lemmaLine = Lemmatizer.Lemmatizer().lemmatizeSingleton(self.lemmatize, line)
                            print(lemmaLine.lemmas + ": ", end='')
                            probs = self.model.classify(lemmaLine)
                            c = 0
                            while c < self.CLASSES:
                                print(str(probs[c]) + " ", end='')
                                c += 1
                            print()
                    elif line[:3] == "---":
                        onTestingData = True
        print("Class 1: Negative, Class 2: Positive")
        f.close()


# entry point for the application
if __name__ == "__main__":
    RottenTomatoesClassifier().run()
    counts = {}
