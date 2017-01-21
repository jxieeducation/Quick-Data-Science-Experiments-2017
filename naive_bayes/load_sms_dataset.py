import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

def load_spam_dataset(path_to_spam="~/Downloads/smsspamcollection/SMSSpamCollection"):
	messages = pd.read_csv(path_to_spam, sep='\t', names=["label", "message"])
	y = np.array([1 if label == "spam" else 0 for label in messages.label])
	vectorizer = CountVectorizer().fit(messages.message)
	X = vectorizer.transform(messages.message)
	vectorizer.vocabulary_inverted = {v: k for k, v in vectorizer.vocabulary_.iteritems()}
	
	### sanity checks ###
	assert X.shape[0] == y.shape[0]
	assert max(vectorizer.vocabulary_inverted.keys()) == X.shape[1] - 1

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
	return  X_train, X_test, y_train, y_test, vectorizer

if __name__ == '__main__':
	X_train, X_test, y_train, y_test, vectorizer = load_spam_dataset()
