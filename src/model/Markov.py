from model import Model, WordProb
import Sentence


# A Markov model of bigrams
class Markov(Model.Model):

    # the total number of bigrams per class is calculated as the probability of a bigram in the class
    # times the number of bigrams in the given sentence
    bigramCounts = [{} for i in range(2)]
    # the total number of bigrams denoms per class is calculated as the probability of a word in the class
    # that isn't at the end of a sentence times the number of words in the given sentence
    bigramDenomsCounts = [{} for i in range(2)]

    # Update the model given a sentence and its probability of
    # belonging to each class
    def update(self, sentence, probs):
        words = sentence.lemmas.split(" ")

        # updates class count and total words
        for i, p in enumerate(probs):
            self.classCounts[i] += p
            self.totalWords[i] += p * len(words)

            previousWord = None
            for j, word in enumerate(words):
                # updates wordCounts for this class and word
                if word in self.wordCounts[i]:
                    self.wordCounts[i][word] = self.wordCounts[i].get(word) + p
                else:
                    self.wordCounts[i][word] = p

                # updates bigramCounts for this class and bigram
                if previousWord is not None:
                    bigram = previousWord + " " + word
                    if bigram in self.bigramCounts[i]:
                        self.bigramCounts[i][bigram] = self.bigramCounts[i].get(bigram) + p
                    else:
                        self.bigramCounts[i][bigram] = p

                # updates bigramDenomsCounts for this class and bigramDenom
                if j != len(words) - 1:
                    if word in self.bigramDenomsCounts[i]:
                        self.bigramDenomsCounts[i][word] = self.bigramDenomsCounts[i].get(word) + p
                    else:
                        self.bigramDenomsCounts[i][word] = p
                previousWord = word

    # Classify a new sentence using the data and a Markov model.
    # Assume every token in the sentence is space-delimited, as the input
    # was.  Return a list of class probabilities.
    def classify(self, sentence):
        probs = []

        # iterates through all classes and calculates a probability proportional to the probability
        # that the sentence belongs to each class
        for i, c in enumerate(self.classCounts):
            # Calculates The probability of the class P(class)
            # which is defined by (# of sentences with class / # of sentences)
            pClass = self.classProbability(i)
            # calculates P(word | class) for all words in the sentence
            pWords = 1
            # the previous word
            previousWord = None
            for j, word in enumerate(sentence.lemmas.split(" ")):
                if j == 0:
                    if word in self.wordCounts[i]:
                        # Adds P(word | class) = wordCount / # of class words
                        pWords *= self.wordCounts[i].get(word) / self.totalWords[i]
                    else:
                        pWords *= self.OUT_OF_VOCAB_PROB
                else:
                    bigram = previousWord + " " + word
                    # if not the first word in the sentence, P(word i | word at i-1) must be multiplied as
                    # well, which is calculated as (# of times bigram appears in the class / # of times
                    # first word of bigram appears in class)
                    if bigram in self.bigramCounts[i]\
                            and previousWord in self.bigramDenomsCounts[i]\
                            and self.bigramDenomsCounts[i].get(previousWord) != 0:
                        pWords *= self.bigramCounts[i].get(bigram) / self.bigramDenomsCounts[i].get(previousWord)
                    else:
                        pWords = self.OUT_OF_VOCAB_PROB
                previousWord = word

            prob = pClass * pWords
            if prob <= 1.1754943508222875e-100:
                probs.append(1.1754943508222875e-38)
            else:
                probs.append(prob)
        return self.normalize(probs)

    # printTopWords: Print n bigrams with the highest
    # Pr(thisClass | bigram) = scale Pr(bigram | thisClass)Pr(thisClass)
    # but skip those that have appeared (in expectation) less than
    # MIN_TO_PRINT times for this class (to avoid random weird bigrams
    # that only show up once in any sentence)
    def printTopWords(self, n):
        for i, c in enumerate(self.bigramCounts):
            print("Cluster " + str(i) + ":")
            wordProbs = []
            for bigram in c.keys():
                if self.bigramCounts[i].get(bigram) >= self.MIN_TO_PRINT:
                    probs = self.classify(Sentence.Sentence(bigram, bigram))
                    wordProbs.append(WordProb.WordProb(bigram, bigram, probs[i]))
            wordProbs.sort(key=lambda x: x.prob, reverse=True)
            j = 0
            while j < n:
                if j >= len(wordProbs):
                    print("No more words...")
                    break
                print(wordProbs[j].word)
                j += 1
