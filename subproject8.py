import numpy as np
from subproject1 import hamiltonian, ndim_Random, dim, N, ndim_Ones
from subproject7 import conjugate_gradient, gram_schmidt, power_method, arnoldi_method, arnoldi_method2


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
    computed_eigenvalue, computed_eigenvector = arnoldi_method2(Q, n=2, tol=1e-5)

    print("largest true eigenvalue:", largest_true_eigenvalue)
    print("computed eigenvalue:", computed_eigenvalue)

    #check if the basis is orthogonal
    assert np.allclose(eigenvectors.T@eigenvectors, np.identity(len(eigenvectors)), atol=1e-5), \
        f"Eigenvectors does not form an orthogonal basis"
    print("Orthogonality test passed")

    # Compare results
    assert np.isclose(np.max(computed_eigenvalue), largest_true_eigenvalue, atol=1e-5), \
        f"Eigenvalue mismatch: expected {largest_true_eigenvalue}, got {np.max(computed_eigenvalue)}"
    print("Result comparison test passed")

    # Check eigenvector direction 
    dot_product = np.abs(np.dot(computed_eigenvector, largest_true_eigenvector))
    assert np.isclose(dot_product, 1.0, atol=1e-5), \
        f"Eigenvector mismatch: computed eigenvector is not aligned with true eigenvector."
    print("Eigenvector direction test passed")

    print("Test passed: Arnoldi method correctly finds the largest eigenvalue and eigenvector.")


test2 = test_arnoldi_method()

#test1 = test_conjugate_gradient()

def test_power_method():
    eigenvalue, eigenvector = power_method(hamiltonian, tol=1e-6)

    # Compute Qv and lambda*v for validation
    Qv = hamiltonian @ eigenvector
    lambda_v = eigenvalue * eigenvector

    # Check if Qv is approximately equal to lambda*v
    assert np.allclose(Qv, lambda_v, atol=1e-6), f"Test failed: Qv={Qv}, lambda*v={lambda_v}"

    print("Test passed: Power method computed the correct eigenvalue and eigenvector.")

# Run the test
test_power_method()
