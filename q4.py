import nltk
from nltk.corpus import wordnet, movie_reviews
from cw1_base import *

class WordNetFeatureExtractor:

	def __init__(self, wordsToConsider):
		self.wordsToConsider = wordsToConsider
		# Pre-compute feature names
		self.containsFeatureNames = [ "contains(%s)" % word for word in self.wordsToConsider ]
		self.containsSynonymFeatureNames = [ "containsSynonymOf(%s)" % word for word in self.wordsToConsider ]
		self.containsHypernymFeatureNames = [ "containsHypernymOf(%s)" % word for word in self.wordsToConsider ]

	def extractFeatures(self, text):
		# Make all words in text lowercase and turn it into a set for faster lookup
		text = set( [ word.lower() for word in text ] )

		featureSet = {}
		for i in range(len(self.wordsToConsider)):
			containsFeature = self.containsFeatureNames[i]
			featureSet[containsFeature] = (self.wordsToConsider[i] in text)

			containsSynonymFeature = self.containsSynonymFeatureNames[i]
			containsHypernymFeature = self.containsHypernymFeatureNames[i]
			wordSynsets = wordnet.synsets( self.wordsToConsider[i] )
			featureSet[containsSynonymFeature] = self.containsSynonymOf(text, wordSynsets)
			featureSet[containsHypernymFeature] = self.containsHypernymOf(text, wordSynsets)
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

def removeRedundantFeatures(words):
	# Cache synsets for all words to retrieval doesn't have to be repeated
	cache = {}
	for word in words:
		cache[word] = wordnet.synsets(word)
	# Only keep words which are NOT similar to each other
	newWords = set(words)
	for wordA in words:
		for wordB in words:
			if wordA == wordB:
				continue # so we don't compare same words
			elif wordSimilarity(cache[wordA], cache[wordB]):
				newWords.remove(wordA)
	return newWords

SIMILARITY_THRESHOLD = 0.75
def wordSimilarity(word1Synsets, word2Synsets):
	# If average similarity between the two word's synsets is above
	# a certain threshold, return zTrue
	sumSimilarity = 0.0
	for synset1 in word1Synsets:
		for synset2 in word2Synsets:
			sim = synset1.wup_similarity(synset2) # returns None if no path between words
			if sim:
				sumSimilarity += sim
	avgSimilarity = sumSimilarity / max(1, (len(word1Synsets) * len(word2Synsets)))
	return (avgSimilarity >= SIMILARITY_THRESHOLD)



if __name__ == "__main__":
	# Load movie review dataset and split it into training and test 
	dataset = getMovieReviewDataset()
	trainingSet, testSet = splitDataset(dataset)
	# Construct frequency distribution of all the corpus' word, after being reduced.
	print "FINDING FEATURES"
	# Use most frequent 2000 words as the features
	wordsToConsider = getWordsToConsider(movie_reviews, 2000)
	featureExtractor = WordNetFeatureExtractor(wordsToConsider)
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