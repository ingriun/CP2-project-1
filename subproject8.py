import numpy as np
from subproject1 import hamiltonian, ndim_Random, dim, N, ndim_Ones
from subproject7 import conjugate_gradient, arnoldi_method

########################### Test functions ##################

def test_conjugate_gradient():
    """Checks if the CG returns indeed the inverse of a function applied to a vector/matrix"""

    print("\n Conjugate Gradient test : ")

    # Define a symmetric matrix
    A = np.array([[4, 1],
                  [2, 5]])
    
    vec = np.array([2, 3])

    def Q(x):
        return np.dot(A,x)
    
    #mat_inv = np.linalg.inv(Q(vec))
    
    x = conjugate_gradient(Q,vec)
    print("Inverse matrix by conjugate gradient", x)

    #print(np.max(np.abs(x - mat_inv)))

    return np.max(np.abs(x))




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

    # Compare results
    assert np.isclose(np.max(computed_eigenvalue), largest_true_eigenvalue, atol=1e-5), \
        f"Eigenvalue mismatch: expected {largest_true_eigenvalue}, got {np.max(computed_eigenvalue)}"

    # Check eigenvector direction 
    dot_product = np.abs(np.dot(computed_eigenvector, largest_true_eigenvector))
    assert np.isclose(dot_product, 1.0, atol=1e-5), \
        "Eigenvector mismatch: computed eigenvector is not aligned with true eigenvector."

    print("Test passed: Arnoldi method correctly finds the largest eigenvalue and eigenvector.")




test1 = test_conjugate_gradient()