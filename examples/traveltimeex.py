if __name__ == "__main__":
	from sys import path
	path.append("../src")

import fastsweeping
import domain
import matplotlib.pyplot as plt
import numpy as np
import potential
import hamiltonian

nx = 300
a1 = -1; a2 = 1; b1 = -1; b2 = 1;
grid = domain.Rectangle(a1,a2,b1,b2,nx)

#centers = np.array([[1,0],[-1,0]], dtype = np.float)
#amps = 0.1*np.ones(2)
#weights = np.ones((2,2))
#Q = potential.gaussian(centers, amps, weights)
#H = hamiltonian.gross_pitaevsky(Q)
H = hamiltonian.eikonal()
R = np.ones(grid.shape)

phi = 20.0*np.ones(grid.shape)
phi[1,1] = 0.0

phi = fastsweeping.travel_times(grid, phi, R, H, sigma = [1.0, 1.0], maxiter = 1000, tol = 1e-10, memory = False, debug = True)

print phi[0,0]
print phi[1,1]
print phi[-2,-2]
print np.sqrt(2.0*(2.0 - 2*grid.h)**2)

plt.imshow(phi[1:-1,1:-1].T, origin = "lower")
plt.colorbar()
plt.show()




