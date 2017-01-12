# principal component regression 
# http://stats.stackexchange.com/questions/82050/principal-component-analysis-and-regression-in-python

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn import cross_validation
import pprint

X, y, coef = make_regression(n_samples=20, n_features=10, n_informative=2, bias=0.3, noise=0.1, coef=True)

pca = PCA()
X_reduced = pca.fit_transform(scale(X))

print "eigen vecs: %s" % str(pca.explained_variance_)
print "pca percents: %s" % str(np.cumsum(pca.explained_variance_ratio_) * 100)

pca_num = []
pca_scores = []

for i in range(0, X_reduced.shape[1]):
	X_subset = X_reduced[:,i] 
	X_subset = X_subset[...,np.newaxis]
	l = LinearRegression()
	l.fit(X_subset, y)
	pca_num += [i]
	pca_scores += [l.score(X_subset, y)]

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(zip(pca_num, pca_scores))

# The results (R^2) are pretty bad! I'd hope that the first principal component does well
# this is not the case probably because it's unsupervised and the principal components (the variance) don't have anything to do with explaining the RSS
