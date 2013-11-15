import sys
import math
import pickle
from cw1_base import *

class BinaryMutualInformationAssociator:

	def __init__(self, classToAssociate):
		self.classToAssociate = classToAssociate
		self.association = {}

	def build(self, allWords, documents):
		# Construct dictionary which maps words to their associate with CLASS ONE
		self.association = { }
		# Compute mutual information for each word
		associationMatrices = self.computeAssociationMatrices(allWords, documents)
		for word in allWords:
			self.association[word] = self.computeMutualInfo( associationMatrices[word] )

	def computeAssociationMatrices(self, allWords, documents):
		# n11 = number of docs with current class that have current word
		# n10 = number of docs with current word that are NOT in the current class
		# n01 = number of docs without current word, but are the current class
		# n00 = number of docs that neither contain the current word nor are the current class		
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
		return associationMatrices

	def computeMutualInfo(self, mat):
		"""Return mutual information for class/word given frequency matrix represents."""
		# Convert frequencies in matrix to floats
		for i in range(2):
			for j in range(2):
				mat[i][j] = float(mat[i][j])
		total = mat[1][1] + mat[1][0] + mat[0][1] + mat[0][0]	
		return (mat[1][1] / total) * math.log( max(1, (total * mat[1][1])) / max(1, ((mat[1][1] + mat[1][0]) * (mat[1][1] + mat[0][1]))), 2 ) + \
			(mat[0][1] / total) * math.log( max(1, (total * mat[0][1])) / max(1, ((mat[0][1] + mat[0][0]) * (mat[1][1] + mat[0][1]))), 2 ) + \
			(mat[1][0] / total) * math.log( max(1, (total * mat[1][0])) / max(1, ((mat[1][1] + mat[1][0]) * (mat[1][0] + mat[0][0]))), 2 ) + \
			(mat[0][0] / total) * math.log( max(1, (total * mat[0][0])) / max(1, ((mat[0][1] + mat[0][0]) * (mat[1][0] + mat[0][0]))), 2 )	

	def select(self, words, numToSelect):
		# Rank words based on their association with chosen class
		rankedWords = sorted(set(words), key = lambda x : self.association.get(x, 0.0), reverse=True) # highest mutual information first	
		return rankedWords[:numToSelect]


if __name__ == "__main__":
	# Parse command line arguments
	if len(sys.argv) < 2:
		sys.exit("Usage: python <outputFilename>".format(sys.argv[0]))
	outputFilename = sys.argv[1]
	# Load movie review dataset
	print "loading movie review dataset..."
	dataset = getMovieReviewDataset()
	allWords = getMovieReviewWords()
	# Determine which features to keep
	print "computing mutual information..."
	associator = BinaryMutualInformationAssociator("pos")
	associator.build(set(allWords), dataset)
	print "outputting to file..."
	with open(outputFilename, "wb") as f:
		pickle.dump(associator.association, f)