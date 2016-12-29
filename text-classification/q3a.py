import sys
from cw1_base import getMovieReviewClassifier, getMovieReviewDataset, removeStopWords

if __name__ == "__main__":
	# Parse command line arguments
	if len(sys.argv) < 3:
		sys.exit("Usage: python {0} <numFeaturesToUse> <filterStopWords>".format(sys.argv[0]))
	numFeatures = int(sys.argv[1])
	filterStopWords = (sys.argv[2] == "true")
	# Load movie review dataset
	dataset = getMovieReviewDataset()
	# Remove all words in stoplist from every document in the dataset
	# We do this here so none of the stopwords can become FEATURES
	# if they're the most frequent (very likely words in the stoplist
	# will also be most frequent as well!)
	if filterStopWords:
		for i in range(len(dataset)):
			dataset[i] = (removeStopWords(dataset[i][0]), dataset[i][1])	

	# Run movie review classifier on two written reviews to see how it handles specific issues
	classifier, trainingSet, testSet = getMovieReviewClassifier(dataset, numFeatures)
	trainingAcc = classifier.test(trainingSet)
	testAcc = classifier.test(testSet)
	print "Training accuracy: %.2f%%" % (trainingAcc * 100)
	print "Test accuracy: %.2f%%" % (testAcc * 100)