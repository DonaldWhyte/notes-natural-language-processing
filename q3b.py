import sys
import pickle
import nltk
from cw1_base import *
from mutual_information import BinaryMutualInformationAssociator

if __name__ == "__main__":
	# Parse command line arguments
	if len(sys.argv) < 3:
		sys.exit("Usage: python {0} <numFeaturesToUse> <filterStopWords> {{<mutualInformationCacheFilename>}}".format(sys.argv[0]))
	numFeatures = int(sys.argv[1])
	filterStopWords = (sys.argv[2] == "true")
	if len(sys.argv) >= 4:
		mutualInformationCacheFilename = sys.argv[3]
	else:
		mutualInformationCacheFilename = None

	# Load movie review dataset
	dataset = getMovieReviewDataset()
	allWords = getMovieReviewWords()
	allClasses = getMovieReviewCategories()
	if filterStopWords:
		for i in range(len(dataset)):
			dataset[i] = (removeStopWords(dataset[i][0]), dataset[i][1])		
	trainingSet, testSet = splitDataset(dataset)

	# Determine which features to keep
	print "DECIDING FEATURES USING MUTUAL INFORMATION"
	# Build associator to rank words and classes
	associator = BinaryMutualInformationAssociator("pos")
	# If cache file was specified, load that and use its values for associator
	if mutualInformationCacheFilename:
		with open(mutualInformationCacheFilename, "rb") as f:
			associator.association = pickle.load(f)
	# Otherwise, manually build the mutual information data on words
	else:
		associator.build(set(allWords), dataset)
	# Select specified number of features and construct feature extractor
	selectedWords = associator.select(allWords, numFeatures)
	featureExtractor = BinomialExtractor(selectedWords)
	# Build and train a classifier which uses a DIFFERENT feature extractor
	print "TRAINING CLASSIFIER"
	classifier = DocumentTypeClassifier(featureExtractor)
	classifier.train(trainingSet)
	trainingAcc = classifier.test(trainingSet)
	testAcc = classifier.test(testSet)
	print "Training accuracy: %.2f%%" % (trainingAcc * 100)
	print "Test accuracy: %.2f%%" % (testAcc * 100)	

	classifier.classifier.show_most_informative_features(30)