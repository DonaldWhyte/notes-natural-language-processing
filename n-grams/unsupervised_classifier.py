import sys
import random
import time

from nltk import FreqDist
from nltk import bigrams
from nltk.corpus import brown

# Constants which determine branching assignment of compound
BRANCH_LABEL_NONE = "-"
BRANCH_LABEL_LEFT = "L"
BRANCH_LABEL_RIGHT = "R"
# Determines the size of the n-grams which are taken compound files
NGRAM_SIZE = 3

class StubLabeller:

	def label(self, compound):
		return BRANCH_LABEL_LEFT

class RandomLabeller:

	def label(self, compound):
		# Generate random number and use it determine branch to use for compound
		# There's a 50% chance of the compoiunt being assigned left and right
		if random.random() < 0.5:
			return BRANCH_LABEL_LEFT
		else:
			return BRANCH_LABEL_RIGHT

def getBracketings(compound):
	return ( compound[0:2], compound[1:3] )

def compoundToString(compound):
	return "-".join(compound)

class MostFrequentLabeller:

	def __init__(self, labelledDataset):
		self.train(labelledDataset)

	def train(self, labelledDataset):
		# Key = bigram (tuple of two words), value = frequency that is the correct branching/compound
		self.correctBracketingFrequencies = {}
		for compound, label in labelledDataset.compounds:
			bracketings = getBracketings(compound)
			if label == BRANCH_LABEL_LEFT:
				correctPair = compoundToString(bracketings[0])
			elif label == BRANCH_LABEL_RIGHT:
				correctPair = compoundToString(bracketings[1])
			else: # NOTE: We ignore unlabelled compounds
				continue
			# Increment or initialise frequency count for
			if correctPair in self.correctBracketingFrequencies:
				self.correctBracketingFrequencies[correctPair] += 1
			else:
				self.correctBracketingFrequencies[correctPair] = 1

	def label(self, compound):
		# Check frequency of both possible bracketings in training dataset
		# Assign most frequenct bracketing
		bracketings = getBracketings(compound)
		leftBracketing = compoundToString(bracketings[0])
		rightBracketing = compoundToString(bracketings[1])
		if self.correctBracketingFrequencies.get(leftBracketing) > \
			self.correctBracketingFrequencies.get(rightBracketing):
			return BRANCH_LABEL_LEFT
		else: # bias introduced here as right branching used if frequencies are equal
			return BRANCH_LABEL_RIGHT

class CorpusBigramLabeller:

	def __init__(self, corpus):
		self.train(corpus)

	def train(self, corpus):
		# Collect all bigrams from the corpus
		corpusBigrams = bigrams(corpus.words())
		# Construct frequency distribution of all bigrams present
		self.bigramFreqDist = FreqDist(corpusBigrams)

	def label(self, compound):
		# Get left and right bracketing of compound
		bracketings = getBracketings(compound)
		# Retrieve frequency of each bracketing as a bigram in the training corpus
		leftFreq = self.bigramFreqDist[bracketings[0][0], bracketings[0][1]]
		rightFreq = self.bigramFreqDist[bracketings[1][0], bracketings[1][1]]
		# Use most frequency
		if leftFreq > rightFreq:
			return BRANCH_LABEL_LEFT
		else: # bias introduced here again
			return BRANCH_LABEL_RIGHT

class CompoundSet:

	def __init__(self, compounds):
		# Construct nested 2-lists to store label for compounds
		self.compounds = [ [compound, BRANCH_LABEL_NONE] for compound in compounds ]

	@classmethod
	def fromFile(cls, filename, compoundSize):
		compounds = [] # list of compounds (lists of words)
		labels = {} # key = index of compound, value = compound's label
		with open(filename, "r") as f:
			for line in f.readlines():
				# Remove trailing newline and split on whitespace to get compound words
				words = line[:-1].split()
				if len(words) >= compoundSize:
					# If there's one more token, treat it as the branch label
					if len(words) >= compoundSize + 1:
						labels[len(compounds)] = words[compoundSize]
					# Only take the first N tokens
					compounds.append( words[:compoundSize] )
		# Construct compound set
		compoundSet = CompoundSet(compounds)
		# Manually label all compounds with labels found (if any)
		for index, label in labels.items():
			compoundSet.setLabel(index, label)
		return compoundSet

	def getLabel(self, index):
		return self.compounds[index][1]

	def setLabel(self, index, newLabel):
		self.compounds[index][1] = newLabel

	def labelAll(self, labeller):
		# Go through all compounds in set and label them using the
		# given labeller object
		for compound in self.compounds:
			compound[1] = labeller.label(compound[0])

	def test(self, labelledSet):
		if len(self.compounds) != len(labelledSet.compounds):
			raise ValueError("Cannot test component set, given labelled set is not the same size")
		# Count the number of correct labels
		total = len(self.compounds)
		correct = 0
		for i in range(total):
			if self.compounds[i][1] == labelledSet.compounds[i][1]:
				correct += 1
		# Compute accuracy
		incorrect = total - correct
		accuracy = float(correct) / float(total)
		# Return test results
		return (correct, incorrect, accuracy)

if __name__ == "__main__":
	# Be sure to seed the random number generator with system time
	random.seed(time.time())

	# Parse command line arguments
	if len(sys.argv) < 3:
		sys.exit("python {0} <unlabelledDataset> <labelledDataset>".format(sys.argv[0]))
	unlabelledDatasetFilename = sys.argv[1]
	labelledDatasetFilename = sys.argv[2]
	# Load both the unlabelled and labelled datasets
	compoundSet = CompoundSet.fromFile(unlabelledDatasetFilename, NGRAM_SIZE)
	labelledSet = CompoundSet.fromFile(labelledDatasetFilename, NGRAM_SIZE)
	# Label dataset
	labeller = MostFrequentLabeller(labelledSet)
	compoundSet.labelAll(labeller)
	# Compare labels with the 'correct' labels in the labelled datasset
	correct, incorrect, accuracy = compoundSet.test(labelledSet)
	# Output test results
	print "{0}/{1} compounds correctly labelled".format(correct, (correct + incorrect))
	print "{0:.2f}% accuracy".format(accuracy * 100)