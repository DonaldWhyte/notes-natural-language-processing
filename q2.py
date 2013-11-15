from cw1_base import getMovieReviewClassifier, getMovieReviewDataset

MOVIE_REVIEWS = [

"""
So this is the first time that Steven Seagal has starred in a Sci-Fi movie. I was blown away by "Sheer Space Seagal" and the amazing performances of every actor involved. I highly recommend you watch this movie.
""",

"""
The original Star Wars trilogy is a masterpiece. It took the world by storm and captivated both children and adults. It was memorable, being talked about for years, and the conclusion to Epsiode VI was outstanding.

It was the end of an era when Episode VI finished. When it was announced that George Lucas would be releasing a prequel trilogy, everyone was excited. What new adventures with the Jedi would happen in the exciting Star Wars universe?

Naturally, there's been a lot of discussion about Episode I. Everyone was queueing up and was buzzing with excitement as they were about to watch the movie.

Like many people, when I left the movie I felt disappointed, however. It was nowhere near the same fantastic quality the original trilogy had. In fact, I think it was a poor movie. Remember that when going into this movie.

"""
]

if __name__ == "__main__":
	# Run movie review classifier on two written reviews to see how it handles specific issues
	classifier = getMovieReviewClassifier(getMovieReviewDataset())[0]
	print "Classified movie review one (overfitting) as: %s" % classifier.classify(MOVIE_REVIEWS[0])
	print "Classified movie review two (discourse) as: %s" % classifier.classify(MOVIE_REVIEWS[1])