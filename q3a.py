import sys
from cw1_base import getMovieReviewClassifier, getMovieReviewDataset

if __name__ == "__main__":
	# Parse command line arguments
	if len(sys.argv) < 2:
		sys.exit("Usage: python {0} <numFeaturesToUse>".format(sys.argv[0]))
	numFeatures = int(sys.argv[1])

	# Run movie review classifier on two written reviews to see how it handles specific issues
	classifier, trainingSet, testSet = getMovieReviewClassifier(getMovieReviewDataset(), numFeatures)
	trainingAcc = classifier.test(trainingSet)
	testAcc = classifier.test(testSet)
	print "Training accuracy: %.2f%%" % (trainingAcc * 100)
	print "Test accuracy: %.2f%%" % (testAcc * 100)