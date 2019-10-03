# the probability and lemma of words/bigrams
class WordProb:
    word = None
    prob = None

    def __init__(self, w, p):
        self.word = w
        self.prob = p

    def __hash__(self):
        return hash(self)

    def __eq__(self, other):
        return self.word == other.word
