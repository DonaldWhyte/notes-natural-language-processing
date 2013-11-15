import sys
import random
import time
import math	
import nltk
from nltk.corpus import movie_reviews, stopwords
from cw1_base import *

class BinaryMutualInformationAssociator:

	def __init__(self, classToAssociate, allWords, documents):
		# n11 = number of docs with current class that have current word
		# n10 = number of docs with current word that are NOT in the current class
		# n01 = number of docs without current word, but are the current class
		# n00 = number of docs that neither contain the current word nor are the current class

		self.classToAssociate = classToAssociate

		# Construct dictionary which maps words to their associate with CLASS ONE
		self.association = { }
		# Construct empty association matrices for each (relative to CLASS ONE!)
		associationMatrices = { }
		for word in allWords:
			associationMatrices[word] = [ [0, 0], [0, 0] ]
		# Count values of matrices incrementally
		for docWords, docClass in documents:
			# Convert to set for O(1) membership test
			docWords = set(docWords)
			# Check if document's class is the one being used for association
			isAssociationClass = (docClass == self.classToAssociate)
			# Now check all words in document and see if they occur in 
			for word in allWords:
				if word in docWords:
					if isAssociationClass:
						associationMatrices[word][1][1] += 1
					else:
						associationMatrices[word][1][0] += 1		
				else:
					if isAssociationClass:
						associationMatrices[word][0][1] += 1
					else:
						associationMatrices[word][0][0] += 1
		# # Compute mutual information from these matrices
		for word in allWords:
			self.association[word] = self.computeMutualInfo( associationMatrices[word] )

	def computeMutualInfo(self, mat):
		"""Return mutual information for class/word given frequency matrix represents."""
		total = mat[1][1] + mat[1][0] + mat[0][1] + mat[0][0]	
		return (mat[1][1] / total) * self.applyLog( (total * mat[1][1]) / max(1, ((mat[1][1] + mat[1][0]) * (mat[1][1] + mat[0][1]))) ) + \
			(mat[0][1] / total) * self.applyLog( (total * mat[0][1]) / max(1, ((mat[0][1] + mat[0][0]) * (mat[1][1] + mat[0][1]))) ) + \
			(mat[1][0] / total) * self.applyLog( (total * mat[1][0]) / max(1, ((mat[1][1] + mat[1][0]) * (mat[1][0] + mat[0][0]))) ) + \
			(mat[0][0] / total) * self.applyLog( (total * mat[0][0]) / max(1, ((mat[0][1] + mat[0][0]) * (mat[1][0] + mat[0][0]))) ) 

	def applyLog(self, value):
		return math.log( max(0.000000000001, logOperand) )

	def select(self, words, numToSelect):
		# Rank words based on their association with chosen class
		rankedWords = sorted(words, key = lambda x : self.association[word], reverse=True)	
		return rankedWords[:numToSelect]

if __name__ == "__main__":
	# Parse command line arguments
	if len(sys.argv) < 3:
		sys.exit("Usage: python {0} <numFeaturesToUse> <filterStopWords>".format(sys.argv[0]))
	numFeatures = int(sys.argv[1])
	filterStopWords = (sys.argv[2] == "true")

	# Load movie review dataset
	dataset = getMovieReviewDataset()
	allWords = getMovieReviewWords()
	allClasses = getMovieReviewCategories()

	trainingSet, testSet = splitDataset(dataset)

	# Determine which features to keep
	print "DECIDING FEATURES"
	# Build associator to rank words and classes
	associator = BinaryMutualInformationAssociator("pos", allWords, dataset)
	# Select specified number of features and construct feature extractor
	selectedWords = associator.select(allWords, NUM_FEATURES)
	featureExtractor = BinomialExtractor(selectedWords)
	# Build and train a classifier which uses a DIFFERENT feature extractor
	print "TRAINING CLASSIFIER"
	classifier = DocumentTypeClassifier(featureExtractor)
	classifier.train(trainingSet)
	print "TESTING CLASSIFIER"
	print "Accuracy: %.2f%%" % (classifier.test(testSet) * 100)
	classifier.classifier.show_most_informative_features(30)