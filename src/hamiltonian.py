import numpy as np

def gross_pitaevsky(Q):
	"""
	Q is a potential function
	"""
	def H(x_1, x_2, xi_1, xi_2):
		"""
			Args:
			x_1 (numpy.ndarray or float): x_1-coordinates on rectangular mesh
			x_2 (numpy.ndarray or float): x_2-coordinates on rectangular mesh
		
			Returns:
			Hamiltonian values (numpy.array) at (x_1, x_2, xi_1, xi_2)
		"""
		return (1.0/(4*np.pi))*np.log(xi_1**2 + xi_2**2) + 0.5*(Q(x_1 + xi_2, x_2 - xi_1) + Q(x_1 - xi_2, x_2 - xi_1))

	return H

def eikonal():
	"""
	"""
	def H(x_1, x_2, xi_1, xi_2):
		return np.sqrt(xi_1**2 + xi_2**2)

	return H

	
