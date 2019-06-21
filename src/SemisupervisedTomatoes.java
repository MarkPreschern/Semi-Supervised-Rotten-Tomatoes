import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;

// Semisupervised Tomatoes:
// EM using Naive Bayes and Markov Models to do sentiment analysis.
//
// Input from train.tsv.zip at
// https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews
//
// itself gathered from Rotten Tomatoes.
//
// Format is PhraseID[unused]   SentenceID  Sentence[tokenized]
//
// Just a few sentiment are provided as this is semisupervised.
//
// We'll only use the first line for each SentenceID, since the others are
// micro-analyzed phrases that would just mess up our counts.
//
// After training, we'll identify the top words for each cluster by
// Pr(cluster | word) - the words that are much more likely in the cluster
// than in the general population - and categorize the new utterances.

public class SemisupervisedTomatoes {

  //The number of classes to train the data to
  private static final int CLASSES = 2;

  // Assume sentence numbering starts with this number in the file
  private static final int FIRST_SENTENCE_NUM = 1;

  // Probability of either a unigram or bigram that hasn't been seen
  // Gotta make this real generous if we're not using logs
  private static final double OUT_OF_VOCAB_PROB = 0.000001;

  // Words to print per class
  private static final int TOP_N = 10;

  // Times (in expectation) that we need to see a word in a cluster
  // before we think it's meaningful enough to print in the summary
  private static final double MIN_TO_PRINT = 15.0;

  // Whether to use the 50 supervised example data or not
  private static boolean SEMISUPERVISED = true;

  //Whether to use a fixed seed when randomizing unsupervised data's classification to start
  private static boolean FIXED_SEED = false;

  //The number of iterations that EM will perform
  private static final int ITERATIONS = 200;

  // characters that should not be analyzed when updating or classifying the model
  private static final String[] STOP_CHARACTERS = {"'", ".", ",", ":)", ":(", "'s", "'d", "'m"};

  // We may play with this in the assignment, but it's good to have common
  // ground to talk about
  private static Random rng = (FIXED_SEED ? new Random(2019) : new Random());

  // whether to use the Naive Bayes model or the Markov model
  private static boolean USE_NAIVE_BAYES = false;

  // the model being used
  private static Model MODEL;

  //Either a NaiveBayesModel or a MarkovModel
  static abstract class Model {
    double[] classCounts;
    double[] totalWords;
    ArrayList<HashMap<String, Double>> wordCounts;

    Model() {
      this.classCounts = new double[CLASSES];
      this.totalWords = new double[CLASSES];
      this.wordCounts = new ArrayList<HashMap<String, Double>>();
      for (int i = 0; i < CLASSES; i++) {
        this.wordCounts.add(new HashMap<String, Double>());
      }
    }

    //Updates the model given a sentence and its probability of belonging to each class
    abstract void update(String sentence, ArrayList<Double> probs);

    //Classifies a new sentence using the data in the model
    abstract ArrayList<Double> classify(String sentence);

    // printTopWords: Print five words with the highest
    // Pr(thisClass | word/bigram) = scale Pr(word/bigram | thisClass)Pr(thisClass)
    // but skip those that have appeared (in expectation) less than
    // MIN_TO_PRINT times for this class (to avoid random weird words
    // that only show up once in any sentence)
    abstract void printTopWords(int n);

    // printTopWords: Print five words with the highest
    // Pr(thisClass | word/bigram) = scale Pr(word/bigram | thisClass)Pr(thisClass)
    // but skip those that have appeared (in expectation) less than
    // MIN_TO_PRINT times for this class (to avoid random weird words
    // that only show up once in any sentence)
    static void printTop(int n, ArrayList<HashMap<String, Double>> counts) {
      for (int c = 0; c < CLASSES; c++) {
        System.out.println("Cluster " + c + ":");
        ArrayList<WordProb> wordProbs = new ArrayList<WordProb>();
        for (String bigram : counts.get(c).keySet()) {
          if (counts.get(c).get(bigram) >= MIN_TO_PRINT) {
            // Treating a word as a one-word sentence lets us use
            // our existing model
            ArrayList<Double> probs = MODEL.classify(bigram);
            wordProbs.add(new WordProb(bigram, probs.get(c)));
          }
        }
        Collections.sort(wordProbs);
        for (int i = 0; i < n; i++) {
          if (i >= wordProbs.size()) {
            System.out.println("No more words...");
            break;
          }
          System.out.println(wordProbs.get(i).word);
        }
      }
    }

    //returns the probability of this class
    double classProbability(int classIndex) {
      double total = 0;
      for (double value : this.classCounts) {
        total += value;
      }
      return this.classCounts[classIndex] / total;
    }

    //adjusts the current proportional probabilities to sum to 1
    ArrayList<Double> normalize(ArrayList<Double> probs) {
      double sum = 0;
      for (Double p : probs) {
        sum += p;
      }
      for (int i = 0; i < probs.size(); i++) {
        probs.set(i, probs.get(i) / sum);
      }
      return probs;
    }

    //removes stop words and makes all words lowercase
    String[] cleanSentence(String sentence) {
      sentence = sentence.toLowerCase();
      ArrayList<String> words = new ArrayList<>(Arrays.asList(sentence.split(" ")));
      for (String stopCharacter : STOP_CHARACTERS) {
        words.remove(stopCharacter);
      }
      return words.toArray(new String[0]);
    }
  }

  // A Naive Bayes model
  static class NaiveBayesModel extends Model {

    NaiveBayesModel() {
      super();
    }

    // Update the model given a sentence and its probability of
    // belonging to each class
    void update(String sentence, ArrayList<Double> probs) {
      String[] words = sentence.split(" ");

      //updates class count and total words
      for (int i = 0; i < probs.size(); i++) {
        this.classCounts[i] += probs.get(i);
        this.totalWords[i] += probs.get(i) * words.length;

        //updates wordCounts for this class and word
        for (String s : words) {
          if (this.wordCounts.get(i).containsKey(s)) {
            this.wordCounts.get(i).put(s, this.wordCounts.get(i).get(s) + probs.get(i));
          } else {
            this.wordCounts.get(i).put(s, probs.get(i));
          }
        }
      }
    }

    // Classify a new sentence using the data and a Naive Bayes model.
    // Assume every token in the sentence is space-delimited, as the input
    // was.  Return a list of class probabilities.
    ArrayList<Double> classify(String sentence) {
      ArrayList<Double> probs = new ArrayList<>(this.classCounts.length);

      //iterates through all classes and calculates a probability proportional to the probability
      //that the sentence belongs to each class
      for (int i = 0; i < this.classCounts.length; i++) {

        //Calculates The probability of the class P(class), which is defined by (# of sentences with class / # of sentences)
        double pClass = this.classProbability(i);
        //calculates P(word | class) for all words in the sentence
        double pWords = 1;
        for (String word : sentence.split(" ")) {
          if (this.wordCounts.get(i).containsKey(word)) {
            //Adds P(word | class) = wordCount / # of class words
            pWords *= this.wordCounts.get(i).get(word) / this.totalWords[i];
          } else {
            pWords *= OUT_OF_VOCAB_PROB;
          }
        }

        //sets the probability of this class to P(class) * P(word | class) for all words which is
        //proportional to the true probability of the class given this sentence
        double prob = pClass * pWords;
        if (Double.compare(prob, 0) == 0) {
          probs.add(i, Double.MIN_NORMAL);
        } else {
          probs.add(i, prob);
        }
      }

      return this.normalize(probs);
    }

    // printTopWords: Print five words with the highest
    // Pr(thisClass | word) = scale Pr(word | thisClass)Pr(thisClass)
    // but skip those that have appeared (in expectation) less than
    // MIN_TO_PRINT times for this class (to avoid random weird words
    // that only show up once in any sentence)
    void printTopWords(int n) {
      this.printTop(n, this.wordCounts);
    }
  }

  // A Markov model
  static class MarkovModel extends Model {
    ArrayList<HashMap<String, Double>> bigramCounts;
    ArrayList<HashMap<String, Double>> bigramDenomsCounts;

    MarkovModel() {
      super();
      this.bigramCounts = new ArrayList<>();
      this.bigramDenomsCounts = new ArrayList<>();
      for (int i = 0; i < CLASSES; i++) {
        this.bigramCounts.add(new HashMap<>());
        this.bigramDenomsCounts.add(new HashMap<>());
      }
    }

    // Update the model given a sentence and its probability of
    // belonging to each class
    void update(String sentence, ArrayList<Double> probs) {
      String[] words = cleanSentence(sentence);

      //updates class count and total words
      for (int i = 0; i < probs.size(); i++) {
        this.classCounts[i] += probs.get(i);
        this.totalWords[i] += probs.get(i) * words.length;

        String previousWord = null;
        for (int j = 0; j < words.length; j++) {
          String s = words[j];

          //updates wordCounts for this class and word
          if (this.wordCounts.get(i).containsKey(s)) {
            this.wordCounts.get(i).put(s, this.wordCounts.get(i).get(s) + probs.get(i));
          } else {
            this.wordCounts.get(i).put(s, probs.get(i));
          }

          //updates bigramCounts for this class and bigram
          if (previousWord != null) {
            String bigram = previousWord + " " + s;
            if (this.bigramCounts.get(i).containsKey(bigram)) {
              this.bigramCounts.get(i).put(bigram, this.bigramCounts.get(i).get(bigram) + probs.get(i));
            } else {
              this.bigramCounts.get(i).put(bigram, probs.get(i));
            }
          }

          //updates bigramDenomsCounts for this class and bigramDenom
          if (j != words.length - 1) {
            if (this.bigramDenomsCounts.get(i).containsKey(s)) {
              this.bigramDenomsCounts.get(i).put(s, this.bigramDenomsCounts.get(i).get(s) + probs.get(i));
            } else {
              this.bigramDenomsCounts.get(i).put(s, probs.get(i));
            }
          }
          previousWord = s;
        }
      }
    }

    // Classify a new sentence using the data and a Markov model.
    // Assume every token in the sentence is space-delimited, as the input
    // was.  Return a list of class probabilities.
    ArrayList<Double> classify(String sentence) {
      ArrayList<Double> probs = new ArrayList<>(this.classCounts.length);

      //iterates through all classes and calculates a probability proportional to the probability
      //that the sentence belongs to each class
      for (int i = 0; i < this.classCounts.length; i++) {

        //Calculates The probability of the class P(class), which is defined by (# of sentences with class / # of sentences)
        double pClass = this.classProbability(i);

        //The probability of this class containing the words in the sentence
        double pWords = 1;

        //the previous word
        String previousWord = null;

        String[] words = cleanSentence(sentence);
        for (int j = 0; j < words.length; j++) {
          String word = words[j];

          if (j == 0) {
            //calculates P(word | class) for all words in the sentence
            if (this.wordCounts.get(i).containsKey(word)) {
              pWords *= this.wordCounts.get(i).get(word) / this.totalWords[i];
            } else {
              pWords *= OUT_OF_VOCAB_PROB;
            }
          } else {
            String bigram = previousWord + " " + word;
            //if not the first word in the sentence, P(word i | word at i-1) must be multiplied as
            //well, which is calculated as (# of times bigram appears in the class / # of times
            //first word of bigram appears in class)
            if (this.bigramCounts.get(i).containsKey(bigram)
                    && this.bigramDenomsCounts.get(i).containsKey(previousWord)
                    && this.bigramDenomsCounts.get(i).get(previousWord) != 0) {
              pWords *= this.bigramCounts.get(i).get(bigram) / this.bigramDenomsCounts.get(i).get(previousWord);
            } else {
              pWords *= OUT_OF_VOCAB_PROB;
            }
          }
          previousWord = word;
        }

        //sets the probability of this class to P(class) * P(word | class) for all words which is
        //proportional to the true probability of the class given this sentence
        double prob = pClass * pWords;
        if (Double.compare(prob, 0) == 0) {
          probs.add(i, Double.MIN_NORMAL);
        } else {
          probs.add(i, prob);
        }
      }

      return this.normalize(probs);
    }

    // printTopWords: Print five words with the highest
    // Pr(thisClass | bigram) = scale Pr(bigram | thisClass)Pr(thisClass)
    // but skip those that have appeared (in expectation) less than
    // MIN_TO_PRINT times for this class (to avoid random weird words
    // that only show up once in any sentence)
    void printTopWords(int n) {
      printTop(n, this.bigramCounts);
    }
  }

  //entry point for the application
  public static void main(String[] args) {
    Scanner myScanner = new Scanner(System.in);
    ArrayList<String> sentences = getTrainingData(myScanner);
    trainModels(sentences);
    MODEL.printTopWords(TOP_N);
    classifySentences(myScanner);
  }

  //parses the training data
  public static ArrayList<String> getTrainingData(Scanner sc) {
    int nextFresh = FIRST_SENTENCE_NUM;
    ArrayList<String> sentences = new ArrayList<String>();
    while (sc.hasNextLine()) {
      String line = sc.nextLine();
      if (line.startsWith("---")) {
        return sentences;
      }
      // Data should be filtered now, so just add it
      sentences.add(line);
    }
    return sentences;
  }

  //Applies the expectation maximization algorithm on the chosen model
  static void trainModels(ArrayList<String> sentences) {
    // We'll start by assigning the sentences to random classes.
    // 1.0 for the random class, 0.0 for everything else
    System.err.println("Initializing models....");
    HashMap<String, ArrayList<Double>> naiveClasses = randomInit(sentences);
    // Initialize the parameters by training as if init were
    // ground truth (essentially starting with M step)
    MODEL = USE_NAIVE_BAYES ? new NaiveBayesModel() : new MarkovModel();
    for (Map.Entry<String, ArrayList<Double>> entry : naiveClasses.entrySet()) {
      MODEL.update(entry.getKey(), entry.getValue());
    }

    //for a set number of iterations, performs the expectation and maximization steps to update the model
    for (int i = 0; i < ITERATIONS; i++) {
      System.err.println("EM round " + i);

      //expectation step
      HashMap<String, ArrayList<Double>> classes = new HashMap<>(sentences.size());
      //determines class probabilities for each sentence
      for (String sentence : sentences) {
        classes.put(sentence, MODEL.classify(sentence));
      }

      //maximization step
      MODEL = USE_NAIVE_BAYES ? new NaiveBayesModel() : new MarkovModel();
      for (Map.Entry<String, ArrayList<Double>> entry : classes.entrySet()) {
        MODEL.update(entry.getKey(), entry.getValue());
      }
    }
  }

  //randomly initializes the unsupervised data based on SEMISUPERVISED, CLASSES, and rng
  static HashMap<String, ArrayList<Double>> randomInit(ArrayList<String> sents) {
    HashMap<String, ArrayList<Double>> counts = new HashMap<String, ArrayList<Double>>();
    for (String sent : sents) {
      ArrayList<Double> probs = new ArrayList<Double>();
      if (SEMISUPERVISED && sent.startsWith(":)")) {
        // Class 1 = positive
        probs.add(0.0);
        probs.add(1.0);
        for (int i = 2; i < CLASSES; i++) {
          probs.add(0.0);
        }
        // Shave off emoticon
        sent = sent.substring(3);
      } else if (SEMISUPERVISED && sent.startsWith(":(")) {
        // Class 0 = negative
        probs.add(1.0);
        probs.add(0.0);
        for (int i = 2; i < CLASSES; i++) {
          probs.add(0.0);
        }
        // Shave off emoticon
        sent = sent.substring(3);
      } else {
        double baseline = 1.0 / CLASSES;
        // Slight deviation to break symmetry
        int randomBumpedClass = rng.nextInt(CLASSES);
        double bump = (1.0 / CLASSES * 0.25);
        if (SEMISUPERVISED) {
          // Symmetry breaking not necessary, already got it
          // from labeled examples
          bump = 0.0;
        }
        for (int i = 0; i < CLASSES; i++) {
          if (i == randomBumpedClass) {
            probs.add(baseline + bump);
          } else {
            probs.add(baseline - bump / (CLASSES - 1));
          }
        }
      }
      counts.put(sent, probs);
    }
    return counts;
  }

  //a comparator for the probability of words/bigrams
  public static class WordProb implements Comparable<WordProb> {
    public String word;
    public Double prob;

    public WordProb(String w, Double p) {
      word = w;
      prob = p;
    }

    //compares the probability of this wordProb and wp
    public int compareTo(WordProb wp) {
      // Reverse order
      if (this.prob > wp.prob) {
        return -1;
      } else if (this.prob < wp.prob) {
        return 1;
      } else {
        return 0;
      }
    }
  }

  //classifies the sentences
  public static void classifySentences(Scanner scan) {
    while (scan.hasNextLine()) {
      String line = scan.nextLine();
      System.out.print(line + ":");
      ArrayList<Double> probs = MODEL.classify(line);
      for (int c = 0; c < CLASSES; c++) {
        System.out.print(probs.get(c) + " ");
      }
      System.out.println();
    }
  }

}