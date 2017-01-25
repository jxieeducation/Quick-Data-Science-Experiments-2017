# https://blog.dbrgn.ch/2013/3/26/perceptrons-in-python/
from random import choice
import numpy as np 

def heavyside_step_function(val):
	res = np.greater_equal(val, np.zeros(val.shape)) * 1
	res[res == 0] = -1
	return res

'''
X is an array of shape (N, dim), last dim is 1 (for \beta_0)
y is an array of shape (N,), with values of -1 or 1
weight is (dim,)

The objective of the perceptron algorithm is to minimize
- \sum_{i \in misclassified} y_i (x_i^T \beta + \beta_0)

The gradient is - \sum_{i \in misclassified} y_i x_i

Therefore we update beta with the negative of the gradient 
'''
class Perceptron:
	def __init__(self, dim):
		self.dim = dim
		self.weight = np.zeros(dim)
		self.train_count = 0

	def getWrongs(self, X, y):
		res = X.dot(self.weight.T)
		res = heavyside_step_function(res)
		wrongs = np.not_equal(res, y)
		return X[wrongs], y[wrongs]

	def fitOne(self, X, y, silent=True, alpha=0.1):
		wrongX, wrongY = self.getWrongs(X, y)
		if not silent:
			print "weights: %s, num wrong: %d" % \
			(str(self.weight), wrongX.shape[0])
		updateX, updateY = choice(zip(wrongX, wrongY))
		# perceptron update formula: w = w + yx (where x_i is misclassified
		self.weight = self.weight + alpha * updateY * updateX
		self.train_count += 1

	def fit(self, X, y, maxit=1000):
		for i in range(maxit):
			try: 
				silent = False if i % 100 == 0 else True
				self.fitOne(X, y, silent=silent)
			except IndexError as e:
				print "converged on iter %d!" % i
				break
		print "final weights are %s" % (str(self.weight))

# note that the last col is the \beta_0 coefficient
X = np.array([
	[-5,-5,1],
	[0,0.2,1],
	[0,-0.6,1],
	[0.1,0.1,1]
]) # m=-1 separating plane
y = np.array([-1, 1, -1, 1])

p = Perceptron(3)
p.fit(X, y)

'''
sample output: 

weights: [ 0.  0.  0.], num wrong: 2
converged on iter 2!
final weights are [ 0.    0.08  0.  ]
'''
