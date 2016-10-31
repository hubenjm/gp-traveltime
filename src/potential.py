import numpy as np

def gaussian(centers, a, weights):
	"""
	Args:
		X (numpy.ndarray): x-coordinates on rectangular mesh
		Y (numpy.ndarray): y-coordinates on rectangular mesh
		centers (numpy.ndarray): Location of centers of Gaussian bumps. Shape = (2, m)
		a (numpy.ndarray): amplitude of each Gaussian bump. Shape = (m,)
		weights (numpy.ndarray): quadratic weights on each exponential. Shape = (2,m)

	Returns:
	potential: function of mesh coordinates (X, Y)
	"""

	assert centers.shape[0] == 2 and weights.shape[0] == 2
	assert a.shape[0] == centers.shape[1] and a.shape[0] == weights.shape[1]
	m = centers.shape[1]

	def Q(X, Y):
		"""
			Args:
			X (numpy.ndarray or float): x-coordinates on rectangular mesh
			Y (numpy.ndarray or float): y-coordinates on rectangular mesh
		
			Returns:
			potential values (numpy.array) on mesh (X, Y)
		"""
		if isinstance(X, (np.ndarray, tuple, list)):
			result = np.zeros_like(X)
			for j in xrange(m):
				result += a[j]*np.exp( -weights[0,j]*(X - centers[0,j])**2 - weights[1,j]*(Y - centers[1,j])**2 )

		else:
			assert np.isscalar(X) and np.isscalar(Y)
			result = 0
			for j in xrange(m):
				result += a[j]*np.exp( -weights[0,j]*(X - centers[0,j])**2 - weights[1,j]*(Y - centers[1,j])**2 )

		return result

	return Q
