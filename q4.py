import random
import time
import nltk
from nltk.corpus import wordnet, movie_reviews
from cw1_base import *

def reduceWords(words):
	"""Using WordNet, return same sized list where each word has been reduced to
	a higher-level word with a similar semantic meaning (to group similar words
	together so they look like the same thing to the classifier)."""
	newWords = []
	for word in words:
		# Get first synset of word and from that, get the root hypernym
		synonymSets = wordnet.synsets(word)
		if len(synonymSets) > 0:
			# Insert all the words from hypernym INTO the new words				
			for synSet in synonymSets:
				newWords += synSet.lemma_names
			#roots = synonymSets[0].root_hypernyms()
			#for root in roots:
			#	newWords += root.lemma_names
		else:
			newWords.append(word)
	return newWords

class WordNetFeatureExtractor:

	def __init__(self, wordsToConsider):
		self.wordsToConsider = wordsToConsider

	def __init__(self, wordsToConsider):
		self.wordsToConsider = wordsToConsider
		# Pre-compute feature names
		self.featureNames = [ "contains(%s)" % word for word in self.wordsToConsider ]

	def extractFeatures(self, text):
		# Make all words in text lowercase and turn it into a set for faster lookup
		text = set( [ word.lower() for word in text ] )
		# Reduce each word to its core word using WordNet
		reducedText = set( reduceWords(text) )
		# Now detemrine which words out of the ones we're considering exist	in reduce text	
		featureSet = {}
		for i in range(len(self.wordsToConsider)):
			key = self.featureNames[i]
			featureSet[key] = (self.wordsToConsider[i] in reducedText)
		return featureSet

		

if __name__ == "__main__":
	# Ensure random number generator has fresh seed
	random.seed(time.time())
	# Load movie review dataset and split it into training and test 
	dataset = getMovieReviewDataset()
	trainingSet, testSet = splitDataset(dataset)
	# Construct frequency distribution of all the corpus' word, after being reduced.
	print "FINDING FEATURES"
	allWords = [ w.lower() for w in movie_reviews.words() ]
	wordFreqDist = nltk.FreqDist( reduceWords(allWords) )
	# Use most frequent 2000 words as the features
	wordsToConsider = wordFreqDist.keys()[:2000]
	featureExtractor = WordNetFeatureExtractor(wordsToConsider)
	# Build and train a classifier which uses a DIFFERENT feature extractor
	print "TRAINING CLASSIFIER"
	classifier = DocumentTypeClassifier(featureExtractor)
	classifier.train(trainingSet)
	print "TESTING CLASSIFIER"
	print "Accuracy: %.2f%%" % (classifier.test(testSet) * 100)
	classifier.classifier.show_most_informative_features(30)