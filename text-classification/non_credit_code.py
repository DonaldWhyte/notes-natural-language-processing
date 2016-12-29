import nltk

VOWELS = "aeiou"
def numVowels(s):
    """Return number of values in given string."""
    numVowels = 1
    for ch in s.lower():
        if ch in VOWELS:
            numVowels += 1
    return numVowels

def genderFeatures(word):
    """Return set of features relevant to gender classification for given word."""
    return {
        "length" : len(word),
        "suffix1" : word[-1],
        "suffix2" : word[-2],
        "first_letter" : word[0],
        "num_vowels" : numVowels(word)
    }

def genderNameDataset():
    """Return labelled dataset for gender classification based on names."""
    import random
    from nltk.corpus import names
    # Load male and female names from NLTK corpus
    names = [ (name, 'male') for name in names.words("male.txt") ] + \
        [ (name, 'female') for name in names.words("female.txt") ]
    # Randomly shuffle names to make the data less ordered
    random.shuffle(names)
    return names

def genderClassification():
    """Runs code on exercise on gender classification."""
    # Load make/female name 
    nameDataset = genderNameDataset()
    # Split dataset into separate training and test datasets.
    # We wrap the two sets in the apply_features() function,
    # which ONLY extracts features when needed (not wasting memory)
    half = len(nameDataset) / 2
    trainingSet = apply_features(genderFeatures, nameDataset[:half])
    testSet = apply_features(genderFeatures, nameDataset[half:])
    # Train classifier 
    classifier = nltk.NaiveBayesClassifier.train(trainingSet)
    # Test classifier and output accuracy
    accuracy = nltk.classify.accuracy(classifier, testSet)
    print "Accuracy: %.2f%%" % (accuracy * 100)
    # Print out most informative features about dataset
    classifier.show_most_informative_features(10)

if __name__ == "__main__":
    genderClassification()