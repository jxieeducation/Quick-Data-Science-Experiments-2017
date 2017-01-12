'''
http://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/GramSchmidt.pdf
'''
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale

X, y, coef = make_regression(n_samples=20, n_features=3, n_informative=2, bias=0.3, noise=0.1, coef=True)
X = scale(X)

print "y is approximated by coef, var(y) is %.2f, var(y_hat) is %.2f" % (np.var(y), np.var(X.dot(coef)))

### now QR factorization ###

q, r = np.linalg.qr(X)
print "q's shape is %s, r's shape is %s" % (str(q.shape), str(r.shape))
print "\n\n"

### properties ###

print "1. Q is X, but with the correlation removed, so the columns are orthogonal"
print np.allclose(0, q[:,0].dot(q[:,1]))
print q.T.dot(q)
print "\n\n"


print "2. X = QR"
print np.allclose(X, q.dot(r))
print "\n\n"


print "3. R is a triangular matrix"
print r
print "The first column of X is q times first column of r"
print "q.dot(r[:,0])"
print np.allclose(q.dot(r[:,0]), X[:,0])
print "\n\n"

print "4. y ~ B1 X"
l1 = LinearRegression()
l1.fit(X, y)
print l1.coef_
print l1.intercept_

l2 = LinearRegression()
l2.fit(scale(q), y)
print l2.coef_
print l2.intercept_
print "these coef should be the same..."
