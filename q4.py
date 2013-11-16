import sys
import nltk
from nltk.corpus import wordnet, movie_reviews
from cw1_base import *

class WordNetFeatureExtractor:

	def __init__(self, wordsToConsider, featureSetToUse):
		self.wordsToConsider = wordsToConsider
		# Determine which features are being considered
		featureSetToUse = featureSetToUse.split("-")
		self.considerWords = ("word" in featureSetToUse)
		self.considerSynonyms = ("syn" in featureSetToUse)
		self.considerHypernyms = ("hyp" in featureSetToUse)
		self.considerAdjectives = ("adj" in featureSetToUse)
		# Pre-compute feature names
		if self.considerWords:
			self.containsFeatureNames = [ "contains(%s)" % word for word in self.wordsToConsider ]
		if self.considerSynonyms:
			self.containsSynonymFeatureNames = [ "containsSynonymOf(%s)" % word for word in self.wordsToConsider ]
		if self.considerHypernyms:
			self.containsHypernymFeatureNames = [ "containsHypernymOf(%s)" % word for word in self.wordsToConsider ]

	def extractFeatures(self, text):
		# Make all words in text lowercase and turn it into a set for faster lookup
		text = set( [ word.lower() for word in text ] )

		featureSet = {}
		for i in range(len(self.wordsToConsider)):
			if self.considerWords:
				containsFeature = self.containsFeatureNames[i]
				featureSet[containsFeature] = (self.wordsToConsider[i] in text)
			# For synonyms and hypernyms, the word's synonym sets are required
			if self.considerSynonyms or self.considerHypernyms:
				wordSynsets = wordnet.synsets( self.wordsToConsider[i] )
			if self.considerSynonyms:
				containsSynonymFeature = self.containsSynonymFeatureNames[i]
				featureSet[containsSynonymFeature] = self.containsSynonymOf(text, wordSynsets)
			if self.considerHypernyms:
				containsHypernymFeature = self.containsHypernymFeatureNames[i]
				featureSet[containsHypernymFeature] = self.containsHypernymOf(text, wordSynsets)

		if self.considerAdjectives:
			for word in text:
				adjectives = self.getAdjectives(word)
				for adj in adjectives:
					try:
						index = self.wordsToConsider.index(adj)
						containsFeature = self.containsFeatureNames[i]
						featureSet[containsFeature] = True
					except ValueError: # if element isn't in words to consider
						pass
		return featureSet

	def containsSynonymOf(self, document, wordSynsets):
		# Check if any of the given word's synonyms are in the document
		synonyms = [ ]
		for synset in wordSynsets:
			for syn in synset.lemma_names:
				if syn in document:
					return True
		return False # if this is reached, no synoynms are present in document

	def containsHypernymOf(self, document, wordSynsets):
		for synset in wordSynsets:
			for hypset in synset.hypernyms():
				for hyp in hypset.lemma_names:
					if hyp in document:
						return True
		return False

if __name__ == "__main__":
	# Parse command line arguments
	if len(sys.argv) < 2:
		usage = "Usage: python {0} <featureSetToUse>".format(sys.argv[0])
		possibleFeatureSets = """Possible Feature Sets:
		word -- Exact word occurrences considered
		syn -- Synonyms of words considered
		hyp -- Hypernyms of words considered
		word-adj -- Get all adjective meanings of words. If one of those adjectives
			   is the same as a word feature, consider that adjective as being in
			   the document (MUST BE USED WITH "word"

		These can be combined using "-". For example, to consider words, synoynms
		and hypernyms, use "word-syn-hyp".
		"""
		sys.exit("{0}\n{1}".format(usage, possibleFeatureSets))
	featureSetToUse = sys.argv[1]

	# Load movie review dataset and split it into training and test 
	dataset = getMovieReviewDataset()
	trainingSet, testSet = splitDataset(dataset)
	# Construct frequency distribution of all the corpus' word, after being reduced.
	print "FINDING FEATURES"
	# Use most frequent 2000 words as the features
	wordsToConsider = getWordsToConsider(movie_reviews, 1000)
	featureExtractor = WordNetFeatureExtractor(wordsToConsider, featureSetToUse)
	#wordsToConsider = removeRedundantFeatures(wordsToConsider)
	# Build and train a classifier which uses a DIFFERENT feature extractor
	print "TRAINING CLASSIFIER"
	classifier = DocumentTypeClassifier(featureExtractor)
	classifier.train(trainingSet)
	trainingAcc = classifier.test(trainingSet)
	testAcc = classifier.test(testSet)
	print "Training accuracy: %.2f%%" % (trainingAcc * 100)
	print "Test accuracy: %.2f%%" % (testAcc * 100)	
	classifier.classifier.show_most_informative_features(30)