import numpy as np
from subproject1 import hamiltonian, ndim_Random, dim, N, ndim_Ones
import numpy.random as random


def conjugate_gradient(Q, b, tol=1e-6, max_iter=100000):
    """ Calculate the inverse of the hamiltonian applied to b

    Parameters:
    Q: function (hamiltonian)
    b: Vector
        Solution of hamiltonian(x) = b
    tol: float
        Convergence tolerance for the method.
    max_iter: int
        Maximum number of iterations.

    Returns:
    vector (ndarray)
        Inverse of the hamiltonian applied to b (x = hamiltonian**(-1)(b))"""
    
    # Initialise x with dimensions of b
    x = 0*ndim_Ones(b.ndim, b.shape[0])

    # Variables inherent to the method
    r_new = b
    r_old = r_new
    p = r_new

    for iteration in range(max_iter):

        A_p = Q(p)
        alpha = np.vdot(r_new, r_new)/(np.vdot(p, A_p))
        x = x + alpha*p
        r_new = r_new - alpha*A_p

        # Check for convergence
        if np.max(np.abs(r_new)) < tol:
            print(f"converged after {iteration+1} iterations.")
            return x

        beta = np.vdot(r_new,r_new)/np.vdot(r_old,r_old)
        p = r_new + beta*p
        r_old = r_new

    # If maximum iterations are reached without convergence, raise an error
    raise RuntimeError(f"Conjugate Gradient failed to converge within {max_iter} iterations.")


def power_method(Q, tol=1e-6, max_iter=100000):
    """define the power method to compute the largest eigenvalue and corresponding eigenvector of an operator Q

    Parameters:
    Q: Function
        The operator (Hamiltonian) to be analysed.
    tol: float
        Convergence tolerance for the method.
    max_iter: int
        Maximum number of iterations.

    Returns:
    tuple
        Largest eigenvalue (float) and corresponding eigenvector (ndarray).    
    """
    #1st step: choose a random v
    v = ndim_Random(dim, N)

    #normalise v to ensure |v|=1
    v =  v/np.linalg.norm(v)

    #initialise eigenvalue approximation
    eigenvalue = None

    for iteration in range(max_iter):
        
        #2nd step: compute w = Qv
        w = Q(v)

        #normalise w to |w| = 1
        w = w/np.linalg.norm(w)

        #approximate the largest eigenvalue
        eigenvalue_new = np.vdot(w, Q(w)).real 

        #check for convergence
        if eigenvalue is not None and np.abs(eigenvalue_new - eigenvalue) < tol:
            print(f"converged after {iteration} iterations.")
            return eigenvalue_new, w
        
        eigenvalue = eigenvalue_new
        v=w #update w for next iteration

 # If maximum iterations are reached without convergence, raise an error
    raise RuntimeError(f"Power method failed to converge within {max_iter} iterations.")






#apply cg to the hamiltonian
"""tol = 1e-6
max_iter = 100000
cg_result = conjugateGradient(hamiltonian, tol, max_iter)

#use cg result as Q in power method
def apply_cg_result(v):
    return np.dot(cg_result, v) """

#inverse_hamiltonian = conjugateGradient(b)

def gram_schmidt(V):
    """Define the gram-schmidt process to orthonormalise the eigenvectors

    Parameters:
    V: ndarray
        Consisting of column vectors that needs to be orthonormalised.

    Returns:
    U: ndarray
        Orthonormalised column vectors    
    """
    n, k = np.shape(V)
    U = np.zeros((n, k))
    U[:, 0] = V[:, 0]/np.linalg.norm(V[:, 0])

    for i in range(1, k):
        U[:, i] = V[:, i]

        for j in range(i):
            U[:, i] = U[:, i] - np.dot(U[:, j], U[:, i]) * U[:, j]

        U[:, i] = V[:, i]/np.linalg.norm(V[:, i])

    return U

def arnoldi_method(Q, n, tol = 1e-6, maxiter = 10000):
    """
    define the arnoldi method to compute the n eigenvalues and corresponding eigenvectors of an operator Q

    Parameters:
    Q: Function
        The operator (Hamiltonian) to be analysed.
    n: int
        Number of eigenvalues
    tol: float
        Convergence tolerance for the method.
    max_iter: int
        Maximum number of iterations.

    Returns:
    tuple
        Largest eigenvalue (float) and corresponding eigenvector (ndarray).
    """
    v = ndim_Random(1, 2) #choosing a random v
    v =  v/np.linalg.norm(v) #normalise v to ensure |v|=1

    K = np.zeros((n, 2)) #initialising the matrix for the Krylov space
    K[0] = v #first element is v

    """K = [Q(K[i-1]) for i in range(1, n+1)]#w_i = Q^i * v
    K = np.array(K)
    print(K)"""
    for index in range(1, n):
        K[index] = Q(K[index-1])
    eigenvalue = None

    for iteration in range(maxiter):
        for i in range(0, n):
            K[i] = Q(K[i]) #compute w_i = Q w_i

        K = gram_schmidt(K) #orthonormalise w
        eigenvalue_new = np.zeros(n)
        for i in range(0, n):
            eigenvalue_new[i] = np.vdot(K[i], Q(K[i])).real #computing eigenvalues
        print(np.shape(eigenvalue_new))

        #check for convergence
        if eigenvalue is not None and np.abs(np.max(eigenvalue_new - eigenvalue)) < tol:
            print(f"converged after {iteration} iterations.")
            return eigenvalue_new, K
        
        #update K and eigenvalue for next step
        K = K
        eigenvalue = eigenvalue_new

    # If maximum iterations are reached without convergence, raise an error
    raise RuntimeError(f"Arnoldi method failed to converge within {maxiter} iterations.")


largest_eigenvalue, eigenvector = arnoldi_method(hamiltonian, n=2, tol=1e-6, maxiter=100000) 
print("Largest eigenvalue:", largest_eigenvalue)
print("Corresponding eigenvector:", eigenvector)

lowest_eigenvalue = 1/largest_eigenvalue

print("lowest eigenvalue:", lowest_eigenvalue)

#run the power method
"""largest_eigenvalue, eigenvector = power_method(inverse_hamiltonian)

print("Largest eigenvalue:", largest_eigenvalue)
print("Corresponding eigenvector:", eigenvector)"""

#calculate the smallest eigenvalue of the inverse of the result
#inverse_matrix = np.linalg.inv(cg_result)
"""smallest_eigenvalue = 1/largest_eigenvalue

print("Smallest eigenvalue:", smallest_eigenvalue)"""


"""A = np.diag([1, 2, 3, 4, 5])

def apply_matrix(v):
    return np.dot(A, v)

computed_eigenvalue, vector = power_method(apply_matrix) 

# Compare with true eigenvalues
true_eigenvalues = np.linalg.eigvals(A)
print("Computed eigenvalues:", computed_eigenvalue)
print("True eigenvalues:", true_eigenvalues)"""