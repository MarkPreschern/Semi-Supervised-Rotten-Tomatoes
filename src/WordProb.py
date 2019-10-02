# the probability and lemma of words/bigrams
class WordProb:
    word = None
    lemma = None
    prob = None

    def __init__(self, w, l, p):
        self.word = w
        self.lemma = l
        self.prob = p
