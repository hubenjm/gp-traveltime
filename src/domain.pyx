import numpy as np
from scipy import sparse
import numpy.matlib
cimport numpy as np
DTYPE = np.float
ctypedef np.float_t DTYPE_t
DTYPEINT = np.int
ctypedef np.int_t DTYPEINT_t

cdef extern from "stdlib.h":
	float fabs(float x)

cimport cython

class Rectangle:

	def __init__(self, a1, a2, b1, b2, nx, debug=False):
		"""
		We view each interior grid point as the center of a given cell.
		Boundary grid points are viewed as being on the actual interfaces of slightly larger cells
		"""
		assert a2 > a1 and b1 < b2 and nx > 0
		self._nx = nx
		self._a1 = a1
		self._a2 = a2
		self._h = float(self._a2-self._a1)/(self._nx-1)

		self._ny = int(round((b2-b1)/self._h)) + 1
		self._b1 = b1
		self._b2 = b1 + (self._ny-1)*self._h #adjusted b2 to make h uniform in either direction

		self._shape = (self._nx, self._ny)
		self._numpoints = self._nx*self._ny
		self._numboundarypoints = 2*(self._nx + self._ny - 2)
		self._X, self._Y = np.meshgrid(np.linspace(self._a1, self._a2, self._nx), np.linspace(self._b1, self._b2, self._ny))
		self._X = self._X.T
		self._Y = self._Y.T

		#for save/logging purposes
		self._domain_params = {'a1': self._a1, 'a2': self._a2, 'b1': self._b1, 'b2': self._b2,
			'h': self._h, 'nx': self._nx, 'ny': self._ny}
	
	@property
	def size(self):
		return self._numpoints

	@property
	def boundary_size(self):
		return self._numboundarypoints
	
	@property
	def shape(self):
		return self._shape

	@property
	def X(self):
		return self._X
	
	@property
	def Y(self):
		return self._Y

	@property
	def h(self):
		return self._h

	@property
	def nx(self):
		return self._nx

	@property
	def ny(self):
		return self._ny
			
	def interior_indices(self, flattened = True):
		"""
		return indices of all interior points
		"""
		xindex = np.matlib.repmat(np.arange(1,self._nx-1), 1, self._ny-2).flatten()
		yindex = np.repeat(np.arange(1,self._ny-1), self._nx-2)
		if flattened:
			return self.convert_to_flattened_indices(xindex, yindex)	
		else:
			return xindex, yindex

	def closure_indices(self, flattened = True):
		xindex = np.matlib.repmat(np.arange(self._nx), 1, self._ny).flatten()
		yindex = np.repeat(np.arange(self._ny), self._nx)
		
		if flattened:
			return self.convert_to_flattened_indices(xindex, yindex)
		else:
			return xindex, yindex

	def boundary_indices(self, skip = 1, flattened = True):
		"""
		return boundary indices as 1D array for use with flattened grid
		counterclockwise order starting with bottom edge
		"""

		bottom = np.arange(0, (self._nx-1)*self._ny+1, skip*self._ny)
		bottom_right_excess = divmod(self._nx - 1, skip)[1]

		right = np.arange((self._nx-1)*self._ny + (skip - bottom_right_excess), self._nx*self._ny, skip)
		right_top_excess = divmod(self._ny - 1 - (skip - bottom_right_excess), skip)[1]

		top = np.arange(self._nx*self._ny - 1 - (skip - right_top_excess)*self._ny, self._ny-2, -skip*self._ny)
		top_left_excess = divmod( self._nx - 1 - (skip - right_top_excess), skip)[1]

		left = np.arange(self._ny - 1 - (skip - top_left_excess), 0, -skip)
		
		boundary_indices = np.hstack((bottom, right, top, left))

		if flattened:
			return boundary_indices
		else:
			return self.convert_from_flattened_indices(boundary_indices)
	
	def on_boundary(self, indices):
		"""
		Given a 1D numpy array of indices or a single int index, checks whether all points are on the boundary of the domain
		"""
		assert isinstance(indices, (np.ndarray, int)), "domain.on_boundary: indices must be an int or np.ndarray of ints"
		sx, sy = self.convert_from_flattened_indices(indices)
		check = (sx == self._nx-1) | (sx == 0) | (sy == self._ny-1) | (sy == 0)		

		if isinstance(indices, int):
			return check
		
		elif isinstance(indices, np.ndarray):
			return check.all()

	def boundary_range(self, numpoints, flattened = True):
		skip = self.boundary_size/numpoints
		assert numpoints <= self.boundary_size
		indices_x, indices_y = self.boundary_to_domain_indices(np.round(np.linspace(0, self.boundary_size - 1, numpoints+1)[:-1]))
		if flattened:
			return self.convert_to_flattened_indices(indices_x, indices_y)
		else:
			return indices_x, indices_y

	def boundary_edge_range(self, edge, numpoints, flattened = True):
		"""
		Returns the array indices for a specific grid edge of the boundary with numpoints regularly spaced points.		
	
		Arguments:
		edge (str): String denoting which edge of domain to discretize ('top', 'right', 'left', or 'bottom')
		numpoints (int): Number of points to return
		flattened (bool): True if indices with respect to the flattened array are desired

		Returns:
		indices if flattened == True, (indices_x, indices_y) otherwise
		"""
		edge_indices = self.boundary_edge(edge, skip = 1, flattened = True)
		edge_indices_x, edge_indices_y = self.convert_from_flattened_indices(edge_indices)
		skip = len(edge_indices)/numpoints
		assert numpoints <= len(edge_indices)

		boundary_indices = self.domain_to_boundary_indices(edge_indices_x, edge_indices_y)
		istart = boundary_indices[0]
		ifinish = boundary_indices[-1]

		indices_x, indices_y = self.boundary_to_domain_indices(np.round(np.linspace(istart, ifinish, numpoints)))
		if flattened:
			return self.convert_to_flattened_indices(indices_x, indices_y)
		else:
			return indices_x, indices_y

	def boundary_edge(self, edge, skip=1, flattened = True):
		"""
		Returns the array indices for a specific grid edge of the boundary

		Arguments:
		edge (str): 'top', 'bottom', 'right', 'left'
		skip (int): Spacing between discrete sample points 
		flattened (bool): True if one desires indices with respect to flattened array
		
		Returns:
		indices if flattened == True, (indices_x, indices_y) otherwise
		"""

		assert edge in ['bottom', 'right', 'top', 'left']

		if flattened:
			if edge == 'bottom':
				return np.arange(0, (self.nx-1)*self.ny+1, skip*self.ny)

			if edge == 'right':
				return np.arange((self.nx-1)*self.ny, self.nx*self.ny, skip)

			if edge == 'top':
				return np.arange(self.nx*self.ny-1, self.ny-2, -skip*self.ny)

			if edge == 'left':
				return np.arange(self.ny-1, -1, -skip)

		else:
			if edge == 'bottom':
				s = np.arange(0, self.nx, skip)
				return s, np.zeros(len(s), dtype = np.int)
			if edge == 'right':
				s = np.arange(0, self.ny, skip)
				return (self.nx-1)*np.ones(len(s), dtype = np.int), s
			
			if edge == 'top':
				s = np.arange(self.nx-1, -1, -skip)
				return s, (self.ny-1)*np.ones(len(s), dtype = np.int)
			
			if edge == 'left':
				s = np.arange(self.ny-1, -1, -skip)
				return np.zeros(len(s), dtype = np.int), s

	def boundary_edges(self, edges, skip=[1,1,1,1], flattened = True):
		"""
		return indices for multiple edges
		edges = [0/1, 0/1, 0/1, 0/1] (bottom, right, top, left)
		"""

		assert len(edges) == 4
		edge_names = ['bottom', 'right', 'top', 'left']
	
		indices = np.array([], dtype = np.int)
		for i in range(4):
			if edges[i] == 1:
				indices = np.append(indices, self.boundary_edge(edge_names[i], skip = skip[i], flattened = True))

		_, unique_idx = np.unique(indices, return_index = True)
	
		if flattened:
			return indices[np.sort(unique_idx)]
		else:
			return self.convert_from_flattened_indices(indices[np.sort(unique_idx)])
				
	def convert_from_flattened_indices(self, index):
		sx, sy = divmod(index, self.ny)
		return sx, sy

	def convert_to_flattened_indices(self, x_index, y_index):
		return y_index + x_index*self.ny

	def border(self, crop, flattened = True):
		"""
		returns indices for a border of width crop inside domain, including the boundary
		"""
		try:
			cx, cy = crop[0], crop[1]
		except:
			print("domain.border: crop must have two integer values")

		#construct 2D indices first
		bottomx = np.tile(np.arange(self.nx), cy)
		bottomy = np.repeat(np.arange(cy), self.nx)

		rightx = np.repeat(np.arange(self.nx-1, self.nx-1-cx, -1), self.ny - 2*cy)
		righty = np.tile(np.arange(cy, self.ny-cy, 1), cx)
	
		topx = np.tile(np.arange(self.nx-1, -1, -1), cy)
		topy = np.repeat(np.arange(self.ny-1, self.ny-1-cy, -1), self.nx)

		leftx = np.repeat(np.arange(cx), self.ny - 2*cy)
		lefty = np.tile(np.arange(self.ny-1-cy, cy-1, -1), cx)

		indexx = np.hstack((bottomx, rightx, topx, leftx))
		indexy = np.hstack((bottomy, righty, topy, lefty))

		if flattened:
			return self.convert_to_flattened_indices(indexx, indexy)
		else:
			return indexx, indexy

	def boundary_to_domain_indices(self, bdindex):
		"""
		Given a set of indices (numpy 1d array of values between 0 and self.boundary_size - 1)
		for point(s) on the boundary of the domain, return a set of 2D grid indices for the 
		corresponding points on the full domain

		input: 1D numpy array of nonnegative integers between 0 and self.boundary_size - 1
		output: Two 1D numpy arrays of nonnegative integers of the same length
		"""

		try:
			n = len(bdindex)
		except:
			bdindex = np.array([bdindex])
			n = 1
		
		assert (bdindex < self.boundary_size).all() and (bdindex >= 0).all()
		xindex = np.zeros(n, dtype=np.int)
		yindex = np.zeros(n, dtype=np.int)

		#bottom edge of domain (left side of matrix) + 2 corners
		mask = bdindex <= self.nx - 1
		xindex[mask] = bdindex[mask]
		yindex[mask] = 0
	
		#right edge of domain (bottom side of matrix) + no corners
		mask = (bdindex >= self.nx) & (bdindex <= self.nx + self.ny - 3)
		xindex[mask] = self.nx-1
		yindex[mask] = bdindex[mask] - self.nx + 1

		#top edge of domain (right side of matrix) + 2 corners
		mask = (bdindex >= self.nx + self.ny - 2) & (bdindex <= 2*self.nx + self.ny - 3)
		xindex[mask] = self.nx - 1 - (bdindex[mask] - self.nx - self.ny + 2)
		yindex[mask] = self.ny-1
	
		#left edge of domain (top side of matrix) + no corners
		mask = (bdindex >= 2*self.nx + self.ny - 2)
		xindex[mask] = 0
		yindex[mask] = self.boundary_size - bdindex[mask]

		if n > 1:	
			return xindex, yindex
		else:
			return np.array([xindex[0], yindex[0]])

	def domain_to_boundary_indices(self, x_index, y_index):
		"""
		Given a set of indices for a vector on the boundary of the domain, return a 1D indices for the corresponding points on the boundary
		input: Two 1D numpy arrays of nonnegative integers of the same length
		output: 1D numpy array of nonnegative integers
		"""
		try:
			n = len(x_index)
		except:
			x_index = np.array([x_index])
			n = 1
		try:
			m = len(y_index)
		except:
			y_index = np.array([y_index])
			m = 1

		assert m == n
		assert ((x_index == self.nx-1) | (x_index == 0) | (y_index == 0) | (y_index == self.ny-1)).all()
		maskleft = x_index == 0
		maskright = x_index == self.nx-1
		maskbottom = y_index == 0
		masktop = y_index == self.ny-1

		bdindex = np.zeros(n, dtype=np.int)
		bdindex[maskleft] = -y_index[maskleft] + self.boundary_size
		bdindex[masktop] = -x_index[masktop] + 2*self.nx + self.ny - 3 
		bdindex[maskright] = y_index[maskright] + self.nx - 1
		bdindex[maskbottom] = x_index[maskbottom]
		
		if n > 1:
			return bdindex
		else:
			return bdindex[0]

	def convert_flattened_to_boundary_indices(self, indices):
		"""
		Given flattened indices for boundary points of grid with respect to ordering of entire set of grid points,
		return a corresponding array of the indices with respect to the boundary ordering
		i.e. return a numpy array of ints between 0 and self.boundary_size - 1
		"""

		# first convert indices to unflattened form
		# then call domain_to_boundary routine
		temp = self.convert_from_flattened_indices(indices)
		return self.domain_to_boundary_indices(temp[0], temp[1])

	def normal_derivative(self, u, corners='average'):
		"""
		Returns a 1D numpy array of shape (grid.numboundarypoints, ) with values equal to du/dn
		We need just u_x on left/right side of domain and u_y on top/bottom
		We use a one-sided difference
		"""
		w = np.zeros(self.shape, dtype=DTYPE)
		
		w[1:-1,  0] = (1.0/self.h)*(u[1:-1, 0] - u[1:-1, 1])
		w[1:-1, -1] = (1.0/self.h)*(u[1:-1,-1] - u[1:-1,-2])
		w[0,  1:-1] = (1.0/self.h)*(u[0, 1:-1] - u[1, 1:-1])
		w[-1, 1:-1] = (1.0/self.h)*(u[-1,1:-1] - u[-2,1:-1])

		#corners
		if corners == 'average':
			w[0, 0] = (0.5/self.h)*(u[0,0] - u[1,0] + u[0,0] - u[0,1])
			w[self.nx-1, 0] = (0.5/self.h)*(u[self.nx-1, 0] - u[self.nx-2, 0] + u[self.nx-1, 0] - u[self.nx-1, 1])
			w[self.nx-1, self.ny-1] = (0.5/self.h)*(u[self.nx-1, self.ny-1] - u[self.nx-2, self.ny-1]\
				+ u[self.nx-1, self.ny-1] - u[self.nx-1, self.ny-2])
			w[0, self.ny-1] = (0.5/self.h)*(u[0, self.ny-1] - u[1, self.ny-1]\
				+ u[0, self.ny-1] - u[0, self.ny-2])
		if corners == 'diagonal':
			h_diagonal = np.sqrt(2)*self.h
			w[0, 0] = (1.0/h_diagonal)*(u[0,0] - u[1,1])
			w[self.nx-1, 0] = (1.0/h_diagonal)*(u[self.nx-1,0] - u[self.nx-2, 1])
			w[self.nx-1, self.ny-1] = (1.0/h_diagonal)*(u[self.nx-1, self.ny-1] - u[self.nx-2, self.ny-2])
			w[0, self.ny-1] = (1.0/h_diagonal)*(u[0, self.ny-1] - u[1, self.ny-2])
		if corners == 'triaverage':
			h_diagonal = np.sqrt(2)*self.h
			w[0, 0] = ( (1.0/self.h)*(u[0,0] - u[1,0]) + (1.0/self.h)*(u[0,0] - u[0,1]) + (1.0/h_diagonal)*(u[0,0] - u[1,1]))/3
			w[self.nx-1, 0] = ( (1.0/self.h)*(u[self.nx-1, 0] - u[self.nx-2, 0]) + 
				(1.0/self.h)*(u[self.nx-1, 0] - u[self.nx-1, 1]) + 
				(1.0/h_diagonal)*(u[self.nx-1,0] - u[self.nx-2, 1]) )/3
			w[self.nx-1, self.ny-1] = ( (1.0/self.h)*(u[self.nx-1, self.ny-1] - u[self.nx-2, self.ny-1]) +
				(1.0/self.h)*(u[self.nx-1, self.ny-1] - u[self.nx-1, self.ny-2]) + 
				(1.0/h_diagonal)*(u[self.nx-1, self.ny-1] - u[self.nx-2, self.ny-2]) )/3
			w[0, self.ny-1] = ( (1.0/self.h)*(u[0, self.ny-1] - u[1, self.ny-1]) +
				(1.0/self.h)*(u[0, self.ny-1] - u[0, self.ny-2]) +
				(1.0/h_diagonal)*(u[0, self.ny-1] - u[1, self.ny-2]) )/3

		if corners == 'zero':
			w[0,0] = 0.0
			w[-1,0] = 0.0
			w[0,-1] = 0.0
			w[-1,-1] = 0.0

		return w.ravel()[self.boundary(flattened=True)]


	#would like to improve upon this gradient function based on fastsweeping method later
	@cython.boundscheck(False)
	def gradient(self, np.ndarray[DTYPE_t, ndim=2] u):
		"""
		input 2-dim float array of nodal values on domain
		returns x and y components of gradient at half grid points (including boundary)
		ux has shape (nx-1) by ny. uy has shape nx by (ny-1)
		"""		
		
		cdef int i, j
		cdef np.ndarray[DTYPE_t, ndim=2] ux = np.zeros([self.nx-1, self.ny], dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=2] uy = np.zeros([self.nx, self.ny-1], dtype=DTYPE)

		for i in xrange(self.nx-1):
			for j in xrange(self.ny):
				ux[i,j] = (u[i+1,j] - u[i,j])/self.h #ux[i,j] = -a_{i+1/2, j}
		
		for i in xrange(self.nx):
			for j in xrange(self.ny-1):
				uy[i,j] = (u[i,j+1] - u[i,j])/self.h #uy[i,j] = -b_{i, j+1/2}

		return ux, uy

	
class Square(Rectangle):
	
	def __init__(self, ll, M, n, debug=False):
		assert M > 0 and n > 0 and len(ll)==2
		Rectangle.__init__(self, ll[0], M + ll[0], ll[1], M + ll[1], n, debug)
		self._n = n
		self._M = M

#	@property
#	def M(self):
#		return self._M

#	@property
#	def n(self):
#		return self._n
