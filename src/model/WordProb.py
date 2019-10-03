# the probability and lemma of words/bigrams
class WordProb:
    word = None
    lemma = None
    prob = None

    def __init__(self, w, l, p):
        self.word = w
        self.lemma = l
        self.prob = p

    def __hash__(self):
        return hash(self.word) * hash(self.lemma)

    def __eq__(self, other):
        return self.word == other.word and self.lemma == other.lemma
