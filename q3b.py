import random
import time
import math	
import nltk
from nltk.corpus import wordnet, movie_reviews, stopwords
from cw1_base import *

ENGLISH_STOPWORDS = set(stopwords.words("english"))
def removeStopWords(words):
	return [ w for w in words if not w in ENGLISH_STOPWORDS ]

class MutualInformationAssociator:

	def __init__(self, allWords, allClasses, documents): # documents = list of tuples (words, cls)
		# n11 = number of docs with current class that have current word
		# n10 = number of docs with current word that are NOT in the current class
		# n01 = number of docs without current word, but are the current class
		# n00 = number of docs that neither contain the current word nor are the current class

		# Create lists of word mutual information values for each class
		self.association = {}
		for cls in allClasses:
			self.association[cls] = {}
		# Look through all classes to maintain dictionaries
		associationMatrices = {}
		for cls in allClasses:
			associationMatrices[cls] = {}
			for word in allWords:
				associationMatrices[cls][word] = [ [0, 0], [0, 0] ]
		# Compute class frequencies
		clsFrequencies = {} # TODO
		# Loop through all the labelled documents
		for docWords, docClass in documents:
			docWords = set(docWords) # convert to set for O(1) membership test
			for word in allWords:
				if word in docWords:
					associationMatrices[docClass][word][1][1] += 1
				else:
					associationMatrices[docClass][word][0][1] += 1
			# TODO: use class frequencues
			associationMatrices[docClass][word][1][0] = (total - associationMatrices[docClass][word][1][1])
			associationMatrices[docClass][word][0][0] = (total - associationMatrices[docClass][word][0][1])

		# TODO: 

		# # Compute mutual information from these counts
		# total = n11 + n10 + n01 + n00
		# divisor = max(1, ((n11 + n10) * (n11 + n01)))
		# logOperand = ((total * n11) / divisor)
		# logOperand = max(0.000000000001, logOperand)
		# mutInfo = (n11 / total) + math.log(logOperand)
		# self.association[cls].append(mutInfo)

	def associate(self, word, cls):
		return self.association[cls].get(word, 0.0)

class FeatureSelector:

	def __init__(self, associator, allClasses):
		self.associator = associator
		self.allClasses = allClasses

	def select(self, words, k):
		perClass = k / len(self.allClasses)
		leftover = k % len(self.allClasses)

		selectedWords = []
		for cls in self.allClasses:
			rankedWords = sorted(words, key = lambda x : self.associator.associate(x, cls))
			if leftover > 0: # also add an extra feature due to upper bound
				upperBound = perClass + 1
				leftover -= 1
			else:
				upperBound = 0
			selectedWords += rankedWords[:upperBound]
		return selectedWords


if __name__ == "__main__":
	# Ensure random number generator has fresh seed
	random.seed(time.time())	
	# Load movie review dataset
	dataset = getMovieReviewDataset()
	allWords = getMovieReviewWords()
	allClasses = getMovieReviewCategories()
	# Remove all words in stoplist from every document in the dataset
	# We do this here so none of the stopwords can become FEATURES
	# if they're the most frequent (very likely words in the stoplist
	# will also be most frequent as well!)
	print "REMOVING STOPWORDS FROM DATASET"
	for i in range(len(dataset)):
		dataset[i] = (removeStopWords(dataset[i][0]), dataset[i][1])
	trainingSet, testSet = splitDataset(dataset)

	# Determine which features to keep
	print "DECIDING FEATURES"
	# Build associator to rank words and classes
	associator = MutualInformationAssociator(allWords, allClasses, dataset)
	# Select specified number of features and construct feature extractor
	selector = FeatureSelector(associator, allClasses)
	selectedWords = selector.select(allWords, NUM_FEATURES)
	featureExtractor = BinomialExtractor(selectedWords)
	# Build and train a classifier which uses a DIFFERENT feature extractor
	print "TRAINING CLASSIFIER"
	classifier = DocumentTypeClassifier(featureExtractor)
	classifier.train(trainingSet)
	print "TESTING CLASSIFIER"
	print "Accuracy: %.2f%%" % (classifier.test(testSet) * 100)
	classifier.classifier.show_most_informative_features(30)