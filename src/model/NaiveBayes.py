from model import Model, WordProb
import Sentence


# A Naive Bayes model
class NaiveBayes(Model.Model):

    # Update the model given a sentence and its probability of
    # belonging to each class
    def update(self, sentence, probs):
        words = sentence.lemmas.split(" ")

        # updates class count and total words
        for i, p in enumerate(probs):
            self.classCounts[i] += p
            self.totalWords[i] += p * len(words)
            # updates wordCounts for this class and word
            for word in words:
                if word in self.wordCounts[i]:
                    self.wordCounts[i][word] = self.wordCounts[i].get(word) + p
                else:
                    self.wordCounts[i][word] = p

    # Classifies a new sentence using the data and a Naive Bayes model.
    # Assume every token in the sentence is space-delimited, as the input
    # was. Return a list of class probabilities.
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
            for word in sentence.lemmas.split(" "):
                if word in self.wordCounts[i]:
                    # Adds P(word | class) = wordCount / # of class words
                    pWords *= self.wordCounts[i].get(word) / self.totalWords[i]
                else:
                    pWords *= self.OUT_OF_VOCAB_PROB

            prob = pClass * pWords
            if prob <= 1.1754943508222875e-100:
                probs.append(1.1754943508222875e-38)
            else:
                probs.append(prob)
        return self.normalize(probs)

    # printTopWords: Print n words with the highest
    # Pr(thisClass | word) = scale Pr(word | thisClass)Pr(thisClass)
    # but skip those that have appeared (in expectation) less than
    # MIN_TO_PRINT times for this class (to avoid random weird words
    # that only show up once in any sentence)
    def printTopWords(self, n):
        for i, c in enumerate(self.wordCounts):
            print("Cluster " + str(i) + ":")
            wordProbs = []
            for word in c.keys():
                if self.wordCounts[i].get(word) >= self.MIN_TO_PRINT:
                    probs = self.classify(Sentence.Sentence(word, word))
                    wordProbs.append(WordProb.WordProb(word, word, probs[i]))
            wordProbs.sort(key=lambda x: x.prob, reverse=True)
            j = 0
            while j < n:
                if j >= len(wordProbs):
                    print("No more words...")
                    break
                print(wordProbs[j].word)
                j += 1
