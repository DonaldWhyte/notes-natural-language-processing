"""Contains code for exercises in chapter 6 of the official NLTK book.
The book, and this chapter, can be found at http://nltk.org/book/ch06.html."""

import random
import time
import nltk
from nltk.classify import apply_features
from nltk.tokenize import regexp_tokenize

MOVIE_REVIEWS = [

"""
So this is the first time that Steven Seagal has starred in a Sci-Fi movie. I was blown away by "Sheer Space Seagal" and the amazing performances of every actor involved.

Don't listen to what other critics have said, they are jealous of this masterpiece.

The actors gave their greatest performances yet. They truly embraced the outstanding story and gritty drama. The special effects are incredible, the humour so subtle and the story so epic.

I highly recommend you watch this movie, for it is the peak of cinema, a true gem. In fact, I think it isn't possible to get better than this - movies have hit their best.

Steven Seagal is an absolute genius in this. I highly recommend you watch this movie.
""",

"""
The original Star Wars trilogy is a masterpiece. It took the world by storm and captivated both children and adults. It was memorable, being talked about for years, and the conclusion to Epsiode VI was outstanding.

It was the end of an era when Episode VI finished. When it was announced that George Lucas would be releasing a prequel trilogy, everyone was excited. What new adventures with the Jedi would happen in the exciting Star Wars universe?

Naturally, there's been a lot of discussion about Episode I. Everyone was queueing up and was buzzing with excitement as they were about to watch the movie.

Like many people, when I left the movie I felt disappointed, however. It was nowhere near the same fantastic quality the original trilogy had. In fact, I think it was a poor movie. Remember that when going into this movie.

"""
]

VOWELS = "aeiou"
def numVowels(s):
    """Return number of values in given string."""
    numVowels = 1
    for ch in s.lower():
        if ch in VOWELS:
            numVowels += 1
    return numVowels

def genderFeatures(word):
    """Return set of features relevant to gender classification for given word."""
    return {
        "length" : len(word),
        "suffix1" : word[-1],
        "suffix2" : word[-2],
        "first_letter" : word[0],
        "num_vowels" : numVowels(word)
    }
    

def genderNameDataset():
    """Return labelled dataset for gender classification based on names."""
    import random
    from nltk.corpus import names
    # Load male and female names from NLTK corpus
    names = [ (name, 'male') for name in names.words("male.txt") ] + \
        [ (name, 'female') for name in names.words("female.txt") ]
    # Randomly shuffle names to make the data less ordered
    random.shuffle(names)
    return names

def genderClassification():
    """Runs code on exercise on gender classification."""
    # Load make/female name 
    nameDataset = genderNameDataset()
    # Split dataset into separate training and test datasets.
    # We wrap the two sets in the apply_features() function,
    # which ONLY extracts features when needed (not wasting memory)
    half = len(nameDataset) / 2
    trainingSet = apply_features(genderFeatures, nameDataset[:half])
    testSet = apply_features(genderFeatures, nameDataset[half:])
    # Train classifier 
    classifier = nltk.NaiveBayesClassifier.train(trainingSet)
    # Test classifier and output accuracy
    accuracy = nltk.classify.accuracy(classifier, testSet)
    print "Accuracy: %.2f%%" % (accuracy * 100)
    # Print out most informative features about dataset
    classifier.show_most_informative_features(10)



class DocumentTypeClassifier:

    TOKENISE_PATTERN = r"(\w+|[\.,;:\$%])"

    def __init__(self, wordsToConsider):
        self.wordsToConsider = wordsToConsider
        # Pre-compute feature names
        self.featureNames = [ "contains(%s)" % word for word in self.wordsToConsider ]
        self.classifier = None
        
    def train(self, trainingSet):
        # Extract features from given dataset and train classifier
        featureSet = apply_features(self.extractPresenceFeatures, trainingSet)
        self.classifier = nltk.NaiveBayesClassifier.train(featureSet)
        
    def test(self, testDataset):
        featureSet = apply_features(self.extractPresenceFeatures, testDataset)
        return nltk.classify.accuracy(self.classifier, featureSet)

    def classify(self, document):
        # Tokenise document by splitting on whitespace and punctuation
        tokens = regexp_tokenize(document, pattern=self.TOKENISE_PATTERN)
        print tokens
        # Extract features from document and return its classification
        featureSet = self.extractPresenceFeatures(tokens)
        return self.classifier.classify(featureSet)
        
    def extractPresenceFeatures(self, text):
        # Make all words in text lowercase and turn it into a set for faster lookup
        text = set( [ word.lower() for word in text ] )
        featureSet = {}
        for i in range(len(self.wordsToConsider)):
            key = self.featureNames[i]
            featureSet[key] = (self.wordsToConsider[i] in text)
        return featureSet
        
    def extractFrequencyFeatures(self, text):
        featureSet = {}
        for word in self.wordsToConsider:
            featureSet[word] = 0 
        for word in text:
            word = word.lower()
            if word in featureSet:
                featureSet[word] += 1
        return featureSet    
    
def getWordsToConsider(corpus, numWordsToUseAsFeatures):
    """'numWordsToCount' most frequent words in given corpus."""
    allWords = nltk.FreqDist( [ w.lower() for w in corpus.words() ] )
    return allWords.keys()[:numWordsToUseAsFeatures]

def getLabelledDatasetFrom(corpus):
    # Generated labelled dataset of documents and what category they belong to
    return [ ( list(corpus.words(fileid)), category)
                  for category in corpus.categories()
                  for fileid in corpus.fileids(category) ]    
    
def documentClassification():
    from nltk.corpus import movie_reviews
    # Get a list of words to consider when extracting features from documents
    wordsToConsider = getWordsToConsider(movie_reviews, 2000)
    # Construct labelled dataset from corpus, splitting it into training and test sets
    dataset = getLabelledDatasetFrom(movie_reviews)
    random.shuffle(dataset) # randomise dataset before splitting
    print "SPLITTING..."
    half = len(dataset) / 2
    trainingSet = dataset[:half]
    testSet = dataset[half:]
    print "SPLIT!"
    # Train and test classifier, printing accuracy
    classifier = DocumentTypeClassifier(wordsToConsider)
    print "STARTING__TRAINING"
    classifier.train(dataset)
    accuracy = classifier.test(dataset)
    print "Accuracy: %.2f%%" % (accuracy * 100)
    classifier.classifier.show_most_informative_features(60)
    # Run classifier on two written reviews to see how it handles specific issues
    print "Classified movie review one (sarcasm) as: %s" % classifier.classify(MOVIE_REVIEWS[0])
    print "Classified movie review two (discourse) as: %s" % classifier.classify(MOVIE_REVIEWS[1])

    return classifier

    
    

if __name__ == "__main__":
    # Be sure to seed random number generator for any computation
    # based on random values
    random.seed(time.time())
    #genderClassification()
    documentClassification()
