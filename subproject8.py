import numpy as np
from subproject1 import hamiltonian, ndim_Random, dim, N, ndim_Ones
from subproject7 import conjugateGradient 

########################### Test functions ##################

def testCG_bis(mat):
    """Checks if the CG returns indeed the inverse of hamiltonian(mat)"""

    x = conjugateGradient(mat, tol=1e-6, max_iter=100000)

    return np.max(np.abs(hamiltonian(x) - mat))


mat = ndim_Random(3,10)

test1 = testCG_bis(mat)
print(test1)