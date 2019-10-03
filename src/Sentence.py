# A sentence is represented as the words in it's original form without stop characters and their lemmas
class Sentence:
    words = None
    lemmas = None

    def __init__(self, words, lemmas):
        self.words = words
        self.lemmas = lemmas

    def __hash__(self):
        return hash(self.words) * hash(self.lemmas)

    def __eq__(self, other):
        return self.words == other.words and self.lemmas == other.lemmas
