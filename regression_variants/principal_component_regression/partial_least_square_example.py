import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression, PLSCanonical
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import pprint
import matplotlib.pyplot as plt

n = 1000
q = 3
p = 10
X = np.random.normal(size=n * p).reshape((n, p))
B = np.array([[1, 2] + [0] * (p - 2)] * q).T
# each Yj = 1*X1 + 2*X2 + noize
Y = np.dot(X, B) + np.random.normal(size=n * q).reshape((n, q)) + 5
X_train = X[0:X.shape[0] / 2]
X_test = X[X.shape[0] / 2:]
Y_train = Y[0:Y.shape[0] / 2]
Y_test = Y[Y.shape[0] / 2:]

# so x1,x2 are useful, x3-10 are bad

pls2 = PLSRegression(n_components=3)
pls2.fit(X_train, Y_train)

print("True B (such that: Y = XB + Err)")
print(B)
# compare pls2.coef_ with B
print("Estimated B")
print(np.round(pls2.coef_, 1))

print "\n\n PLS scored: %.2f" % pls2.score(X_test, Y_test)

# high variance and have high correlation with the response, in contrast to principal components regression which keys only on high variance

# https://github.com/scikit-learn/scikit-learn/blob/14031f6/sklearn/cross_decomposition/pls_.py#L295
# this is the weight estimation step, note: Yk's k is the iteration / component #

# blah, not going to dig into the NIPALS algorithm

########################################################################

pca = PCA()
X_train_reduced = pca.fit_transform(scale(X_train))[:,0:5] # take top 5 dim of pca
l = LinearRegression()
l.fit(X_train_reduced, Y_train)

print l.score(pca.transform(scale(X_test))[:,0:5], Y_test)

### this basically shows the cons of PCR. Maximizing the variance of the covariates doesn't help if they are not related to the output

