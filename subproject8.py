import numpy as np
from subproject1 import hamiltonian, ndim_Random, dim, N, ndim_Ones
from subproject7 import conjugateGradient, arnoldi_method

########################### Test functions ##################

def testCG_bis(mat):
    """Checks if the CG returns indeed the inverse of hamiltonian(mat)"""

    x = conjugateGradient(mat, tol=1e-6, max_iter=100000)

    return np.max(np.abs(hamiltonian(x) - mat))




def test_arnoldi_method():
    # Define a simple symmetric matrix 
    A = np.array([
        [4, 1],
        [1, 3]
    ])

    def Q(x):
        return np.matmul(A,x) #np.dot(A,x)  

    # True eigenvalues and eigenvectors of A
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    largest_true_eigenvalue = eigenvalues[-1]
    largest_true_eigenvector = eigenvectors[:, -1]

    # Run the Arnoldi method
    computed_eigenvalue, computed_eigenvector = arnoldi_method(Q, n=1, tol=1e-5)

    #check if the basis is orthogonal
    assert np.allclose(eigenvectors.T@eigenvectors, np.identity(len(eigenvectors)), atol=1e-5), \
        f"Eigenvectors does not form an orthogonal basis"
    print("Orthogonality test passed")

    # Compare results
    assert np.isclose(np.max(computed_eigenvalue), largest_true_eigenvalue, atol=1e-5), \
        f"Eigenvalue mismatch: expected {largest_true_eigenvalue}, got {np.max(computed_eigenvalue)}"
    print("Result comparison test passed")

    # Check eigenvector direction (up to sign ambiguity)
    dot_product = np.abs(np.dot(computed_eigenvector, largest_true_eigenvector))
    assert np.isclose(dot_product, 1.0, atol=1e-5), \
        "Eigenvector mismatch: computed eigenvector is not aligned with true eigenvector."
    print("Eigenvector direction test passed")

    print("Test passed: Arnoldi method correctly finds the largest eigenvalue and eigenvector.")


# Run the test
test_arnoldi_method()

mat = ndim_Random(3,10)

test1 = testCG_bis(mat)
print(test1)