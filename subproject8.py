import numpy as np
import matplotlib.pyplot as plt
from subproject1 import ndim_Random, dim, N
from subproject7 import conjugate_gradient, arnoldi_method, gram_schmidt


########################### Tests on Conjugate Gradient ######################

def test_conjugate_gradient(A, b):
    """Checks if the CG returns indeed the inverse of a function applied to a vector/matrix"""

    def Q(x):
        return np.matmul(A,x)
    
    x_exact = np.linalg.solve(A,b)
    print("\n Exact inverse matrix : \n ", x_exact, "\n")
    
    x_cg = conjugate_gradient(Q,b)
    print("Inverse matrix by conjugate gradient : \n ", x_cg, "\n")

    print("Convergence of the two matrices (tolerance = 1e-6): ", np.allclose(x_exact, x_cg, atol=1e-6), "\n")


def simple_matrix_CG():

    print("CG with a simple matrix : \n")

    A = np.array([[4, 1],
                  [1, 3]])
    print("A : \n", A)
    
    b = np.array([1, 2])
    print("b : ", b)

    test_conjugate_gradient(A, b)


def identity_matrix_CG():
    
    print("CG with the identity matrix : \n")

    A = np.eye(3)
    print("A : \n", A)

    b = np.array([1, 2, 3])
    print("b : ", b)

    test_conjugate_gradient(A, b)


def diag_matrix_CG():

    print("CG with a diagonal matrix : \n")

    A = np.diag([1, 2, 3, 4])
    print("A : \n", A)

    b = np.array([2, 1, 5, 3])
    print("b : ", b)

    test_conjugate_gradient(A, b)

    
def large_matrix_CG():
    
    print("CG with a large matrix : \n")

    A = np.array([[5, 7, 6, 5],
                  [7, 10, 8, 7],
                  [6, 8, 10,9],
                  [5, 7, 9, 10]])
    print("A : \n", A)

    b = ndim_Random(1,4)
    print("b : \n", b)

    test_conjugate_gradient(A, b)



########################### Test on the Arnoldi Method #####################


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
    computed_eigenvalue, computed_eigenvector = arnoldi_method(Q, n=1,N=2, tol=1e-5)
    print("Computed eigenvalue : \n", computed_eigenvalue[0])
    print("Expected eigenvalue : \n",largest_true_eigenvalue)

    #check if the basis is orthogonal
    assert np.allclose(eigenvectors.T@eigenvectors, np.identity(len(eigenvectors)), atol=1e-5), \
        f"Eigenvectors does not form an orthogonal basis"
    print("Orthogonality test passed")

    # Compare results
    assert np.isclose(np.max(computed_eigenvalue), largest_true_eigenvalue, atol=1e-5), \
        f"Eigenvalue mismatch: expected {largest_true_eigenvalue}, got {np.max(computed_eigenvalue)}"
    print("Result comparison test passed")
    print(f'computed eigenvector: {computed_eigenvector}')
    print(f'true eigenvector: {largest_true_eigenvector}')
    # Check eigenvector direction (up to sign ambiguity)
    dot_product = np.abs(np.dot(computed_eigenvector, largest_true_eigenvector))
    print(f'dotproduct: {dot_product}')
    assert np.isclose(dot_product, 1.0, atol=1e-5).all(), \
        "Eigenvector mismatch: computed eigenvector is not aligned with true eigenvector."
    print("Eigenvector direction test passed")

    print("Test passed: Arnoldi method correctly finds the largest eigenvalue and eigenvector.")


############################ Test on Gram-Schmidt ###################

def test_gram_schmidt():
    # Define some sample input vectors (linearly independent)
    V = np.array([[1, 0, 0],
                  [1, 1, 0],
                  [1, 1, 1]], dtype=float)

    # Call the Gram-Schmidt function
    U = gram_schmidt(V)

    # Check orthonormality
    k = U.shape[1]
    for i in range(k):
        # Check if each vector has unit norm
        assert np.isclose(np.linalg.norm(U[:, i]), 1.0), f"Vector {i} is not normalized."

        for j in range(i + 1, k):
            # Check if vectors are orthogonal
            assert np.isclose(np.dot(U[:, i], U[:, j]), 0.0), f"Vectors {i} and {j} are not orthogonal."

    print("All tests passed for gram_schmidt!")



################################ Tests calls ##############################

# On Conjugate Gradient 
print("############## Conjugate Gardient tests ############ \n")
simple_matrix_CG()
print("#################################################### \n")
identity_matrix_CG()
print("#################################################### \n")
diag_matrix_CG()
print("#################################################### \n")
large_matrix_CG()

# On Gram-Schmidt
print("################ Gram-Schmidt test ################# \n")
test_gram_schmidt()

print("\n")

# On Arnoldi
print("############### Arnoldi Method test ################ \n")
test_arnoldi_method()


