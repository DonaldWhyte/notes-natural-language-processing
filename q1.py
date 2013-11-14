from cw1_base import getMovieReviewClassifier, getMovieReviewDataset

if __name__ == "__main__":
	# Find 30 most informative features for classifying movie reviews as positive or negative
	classifier = getMovieReviewClassifier(getMovieReviewDataset())[0]
	classifier.classifier.show_most_informative_features(30)