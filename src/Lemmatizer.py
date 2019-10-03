import Sentence
import nltk
from nltk.corpus import wordnet


# Represents a lemmatizer, which reduces inflectional forms and sometimes derivationally related forms of a word to a
# common base form, with the use of a vocabulary and morphological analysis of words
class Lemmatizer:

    # Maps the list of sentences to a list of lemmatized sentences
    def lemmatize(self, lemmatize, sentences):
        if lemmatize:
            print("Lemmatizing Sentences...")
        lemmatizer = nltk.WordNetLemmatizer()
        mappedSentences = []
        for i, sentence in enumerate(sentences):
            sentence = self.removeStopCharacters(sentence)
            if lemmatize:
                if i % 100 == 0 and i != 0:
                    print(str(i) + "/" + str(len(sentences) - 1) + " sentences lemmatized")
                lemmas = self.getLemmas(lemmatizer, sentence)
                mappedSentences.append(Sentence.Sentence(sentence, lemmas)) # maps sentence to lemmas
            else:
                mappedSentences.append(Sentence.Sentence(sentence, sentence)) # maps sentence to sentence
        return mappedSentences

    # Maps the sentence to it's lemma
    def lemmatizeSingleton(self, lemmatize, sentence):
        sentence = self.removeStopCharacters(sentence)
        if lemmatize:
            return Sentence.Sentence(sentence, self.getLemmas(nltk.WordNetLemmatizer(), sentence))
        else:
            return Sentence.Sentence(sentence, sentence)

    # returns the sentence without characters that are deemed unnecessary
    @staticmethod
    def removeStopCharacters(sentence):
        stopCharacters = ['.', ',', '!', '?', '\"', '\'', '[', ']', ';', '~', '\\', '/', '`']
        for stopCharacter in stopCharacters:
            sentence = sentence.replace(stopCharacter, "")
        sentence = sentence.replace("  ", " ")  # removes extra whitespace
        sentence = sentence.strip() # removes trailing and leading whitespace
        sentence = sentence.lower() # makes the sentence all lowercase
        return sentence

    # derives the lemma for each respective word and part of speech, and builds a space-separated sentence with them
    def getLemmas(self, lemmatizer, sentence):
        return " ".join([lemmatizer.lemmatize(w, self.get_wordnet_pos(w)) for w in nltk.word_tokenize(sentence)])

    # Map POS tag to first character lemmatize() accepts
    @staticmethod
    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
