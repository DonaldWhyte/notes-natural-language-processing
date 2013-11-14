"""Contains code for exercises in chapter 6 of the official NLTK book.
The book, and this chapter, can be found at http://nltk.org/book/ch06.html."""

import random
import time
import nltk
from nltk.classify import apply_features
from nltk.tokenize import regexp_tokenize
from nltk.corpus import movie_reviews

class BinomialExtractor:

    def __init__(self, wordsToConsider):
        self.wordsToConsider = wordsToConsider
        # Pre-compute feature names
        self.featureNames = [ "contains(%s)" % word for word in self.wordsToConsider ]

    def extractFeatures(self, text):
        # Make all words in text lowercase and turn it into a set for faster lookup
        text = set( [ word.lower() for word in text ] )
        featureSet = {}
        for i in range(len(self.wordsToConsider)):
            key = self.featureNames[i]
            featureSet[key] = (self.wordsToConsider[i] in text)
        return featureSet
   
class MultinomialExtractor:

    def __init__(self, wordsToConsider):
        self.wordsToConsider = wordsToConsider

    def extractFeatures(self, text):
        featureSet = {}
        for word in self.wordsToConsider:
            featureSet[word] = 0 
        for word in text:
            word = word.lower()
            if word in featureSet:
                featureSet[word] += 1
        return featureSet    

class DocumentTypeClassifier:

    TOKENISE_PATTERN = r"(\w+|[\.,;:\$%])"

    def __init__(self, featureExtractor):
        self.featureExtractor = featureExtractor
        self.classifier = None
        
    def train(self, trainingSet):
        # Extract features from given dataset and train classifier
        featureSet = apply_features(self.featureExtractor.extractFeatures, trainingSet)
        self.classifier = nltk.NaiveBayesClassifier.train(featureSet)
        
    def test(self, testDataset):
        featureSet = apply_features(self.featureExtractor.extractFeatures, testDataset)
        return nltk.classify.accuracy(self.classifier, featureSet)

    def classify(self, document):
        # Tokenise document by splitting on whitespace and punctuation
        tokens = regexp_tokenize(document, pattern=self.TOKENISE_PATTERN)
        # Extract features from document and return its classification
        featureSet = self.featureExtractor.extractFeatures(tokens)
        return self.classifier.classify(featureSet)
    
def getMovieReviewWords():
    return movie_reviews.words()

def getMovieReviewCategories():
    return movie_reviews.categories()

def getWordsToConsider(corpus, numWordsToUseAsFeatures):
    """'numWordsToCount' most frequent words in given corpus."""
    allWords = nltk.FreqDist( [ w.lower() for w in corpus.words() ] )
    return allWords.keys()[:numWordsToUseAsFeatures]

def getLabelledDatasetFrom(corpus):
    # Generated labelled dataset of documents and what category they belong to
    return [ ( list(corpus.words(fileid)), category)
                  for category in corpus.categories()
                  for fileid in corpus.fileids(category) ]    
    
def getMovieReviewDataset():
    return getLabelledDatasetFrom(movie_reviews)

def getMovieReviewClassifier(dataset):
    # Ensure random number generator has fresh seed
    random.seed(time.time())
    # Get a list of words to consider when extracting features from documents
    wordsToConsider = getWordsToConsider(movie_reviews, 2000)
    # Construct labelled dataset from corpus, splitting it into training and test sets
    random.shuffle(dataset) # randomise datsaset before splitting
    half = len(dataset) / 2
    trainingSet = dataset[:half]
    testSet = dataset[half:]
    # Train classifier and return it
    featureExtractor = BinomialExtractor(wordsToConsider) # use binomial word features
    classifier = DocumentTypeClassifier(featureExtractor)
    classifier.train(trainingSet)
    return (classifier, trainingSet, testSet)

if __name__ == "__main__":
    classifier, trainingSet, testSet = getMovieReviewClassifier( getMovieReviewDataset() )
    accuracy = classifier.test(testSet)
    print "Accuracy: %.2f%%" % (accuracy * 100)    