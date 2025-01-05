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
    A_p = Q(p)

    for iteration in range(max_iter):

        A_p = Q(p)
        alpha = np.vdot(r_new, r_new)/(np.vdot(p, A_p))
        x = x + alpha*p
        r_new = r_new - alpha*A_p

        # Check for convergence
        if np.max(np.abs(r_new)) < tol:
            #print(f"converged after {iteration+1} iterations.")
            return x

        beta = np.vdot(r_new,r_new)/np.vdot(r_old,r_old)
        p = r_new + beta*p
        r_old = r_new

    # If maximum iterations are reached without convergence, raise an error
    raise RuntimeError(f"Conjugate Gradient failed to converge within {max_iter} iterations.")

def hinv(b):
    return conjugate_gradient(hamiltonian,b)

def power_method(Q, tol=1e-6, max_iter=100000):

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


def gram_schmidt(V):
    """Define the gram-schmidt process to orthonormalise the eigenvectors

    Parameters:
    V: ndarray
        Consisting of column vectors that needs to be orthonormalised.

    Returns:
    U: ndarray
        Orthonormalised column vectors"""    
    
    U = np.zeros(np.shape(V), dtype=complex)
    U[0, :] = V[0, :]/np.linalg.norm(V[0, :])

    for i in range(1, np.shape(V)[0]):
        U[i, :] = V[i, :]

        for j in range(i):
            U[i, :] = U[i, :] - np.dot(U[j, :], U[i, :]) * U[j, :]

        U[i, :] = U[i, :]/np.linalg.norm(U[i, :])

    return U

def arnoldi_method(Q, n, tol = 1e-6, max_iter = 10000):
    
    """define the arnoldi method to compute the n eigenvalues and corresponding eigenvectors of an operator Q

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
    v = ndim_Random(dim,N) #choosing a random v
    v =  v/np.linalg.norm(v) #normalise v to ensure |v|=1
    print(v)

    K = np.zeros((dim, N, N), dtype = complex) #initialising the matrix for the Krylov space

    K[0] = v #first element is v
    print(K)
    # Krylov subspace as a list of vectors
    #K = [v]

    K = [Q(K[i-1]) for i in range(1, n)]#w_i = Q^i * v
    print(K)
    K = np.array(K)
    print(K)

    #for index in range(1, n):
    #    K[index] = Q(K[index-1])
    eigenvalue = None

    for iteration in range(max_iter):
        for i in range(0, n):
            K[i] = Q(K[i]) #compute w_i_new = Q * w_i

        K = gram_schmidt(K) #orthonormalise w

        eigenvalue_new = np.zeros(n)
        for i in range(0, n):
            eigenvalue_new[i] = np.vdot(K[i], Q(K[i])).real #computing eigenvalues
        print(np.shape(eigenvalue_new))

        #check for convergence
        if eigenvalue is not None and np.abs(np.max(eigenvalue_new - eigenvalue)) < tol:
            print(f'converged after {iteration} iterations.')
            return eigenvalue_new, K
        
        #update K and eigenvalue for next step
        K = K
        eigenvalue = eigenvalue_new

    # If maximum iterations are reached without convergence, raise an error
    raise RuntimeError(f'Arnoldi method failed to converge within {max_iter} iterations.')

def arnoldi_method3(Q, n, tol = 1e-6, max_iter = 10000):
    
    """define the arnoldi method to compute the n eigenvalues and corresponding eigenvectors of an operator Q

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
    v = np.random.randn(Q(0).shape[0]) #choosing a random v
    print(v)
    v =  v/np.linalg.norm(v) #normalise v to ensure |v|=1
    print(np.linalg.norm(v[0]))
    print(f'v = {v}')

    w = [v] #initialising Krylov subspace
    print(w)

    for i in range(1, n):
        w.append(Q(w[i-1]))
    
    K = np.array(w)
    print(K)

    eigenvalue = None

    for iteration in range(max_iter):

        for i in range(0, n):
            K[i] = Q(K[i]) #compute w_i_new = Q * w_i
        K = gram_schmidt(K)

        eigenvalue_new = np.zeros(n)
        for i in range(0, n):
            eigenvalue_new[i] = np.vdot(K[i], Q(K[i])).real #computing eigenvalues

        #check for convergence
        if eigenvalue is not None and np.abs(np.max(eigenvalue_new - eigenvalue)) < tol:
            print(f'converged after {iteration+1} iterations.')
            return eigenvalue_new, K
        
        #update K and eigenvalue for next step
        K = K
        eigenvalue = eigenvalue_new

    # If maximum iterations are reached without convergence, raise an error
    raise RuntimeError(f'Arnoldi method failed to converge within {max_iter} iterations.')  



def arnoldi_method2(Q, n, tol=1e-6, max_iter=10000):
    
    """Compute the n eigenvalues and eigenvectors of an operator Q using the Arnoldi method.
    
    Parameters:
    Q: function
        The operator (e.g., Hamiltonian) applied to vectors.
    n: int
        Number of eigenvalues to compute.
    tol: float
        Convergence tolerance.
    maxiter: int
        Maximum number of iterations.

    Returns:
    eigenvalues: ndarray
        The computed eigenvalues.
    eigenvectors: list
        List of corresponding eigenvectors (multidimensional arrays).
    """
    # Initialize the first vector in the Krylov subspace
    v = ndim_Random(dim, N)
    v = v/np.linalg.norm(v)  # Normalize to ensure |v| = 1

    # Krylov subspace as a list of vectors
    K = [v]

    # Hessenberg matrix
    H = np.zeros((n + 1, n), dtype=complex)

    for k in range(n):
        # Compute w = Q * v_k
        w = Q(K[-1])

        # Arnoldi orthogonalization
        for j in range(len(K)):
            H[j, k] = np.vdot(K[j], w)  # Inner product
            w -= H[j, k] * K[j]  # Remove component along K[j]

        H[k + 1, k] = np.linalg.norm(w)  # Residual norm
        if H[k + 1, k] < tol:
            print(f"Converged at step {k + 1}")
            break

        K.append(w / H[k + 1, k])  # Normalize and append new basis vector

    # Compute eigenvalues and eigenvectors of the Hessenberg matrix
    eigenvalues, eigvecs = np.linalg.eig(H[:n, :n])

    # Transform eigenvectors back to the original space
    eigenvectors = [sum(eigvecs[j, i] * K[j] for j in range(n)) for i in range(n)]

    return eigenvalues, eigenvectors


def arnoldi_method4(Q, n, N, tol = 1e-6, max_iter = 10000):
    
    """define the arnoldi method to compute the n eigenvalues and corresponding eigenvectors of an operator Q

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
    v = ndim_Random(dim,N) #choosing a random v
    v =  v/np.linalg.norm(v) #normalise v to ensure |v|=1


    K = np.zeros((n, N), dtype = complex) #initialising the matrix for the Krylov space

    K = np.array([v for i in range(0,n)])

    for index in range(1,n):
        for num in range(0,index):
            K[index] = Q(K[index])

    eigenvalue = None

    for iteration in range(max_iter):
        for i in range(0, n):
            K[i] = Q(K[i]) #compute w_i_new = Q * w_i

        K = gram_schmidt(K) #orthonormalise w

        eigenvalue_new = np.zeros(n)
        for i in range(0, n):
            eigenvalue_new[i] = np.vdot(K[i], Q(K[i])).real #computing eigenvalues
        #print(np.shape(eigenvalue_new))

        #check for convergence
        if eigenvalue is not None and np.abs(np.max(eigenvalue_new - eigenvalue)) < tol:
            print(f'converged after {iteration} iterations.')
            return eigenvalue_new, K
        
        #update K and eigenvalue for next step
        K = K
        eigenvalue = eigenvalue_new

    # If maximum iterations are reached without convergence, raise an error
    raise RuntimeError(f'Arnoldi method failed to converge within {max_iter} iterations.')

def arnoldi_method5(Q, n, tol = 1e-5, max_iter = 10000):
    
    """define the arnoldi method to compute the n eigenvalues and corresponding eigenvectors of an operator Q

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
    print(v)

    K = np.zeros((n, len(v)), dtype = complex) #initialising the matrix for the Krylov space
    #print(K)
    K[0] = v #first element is v
    print(K)
    # Krylov subspace as a list of vectors
    #K = [v]

    """K = [Q(K[i-1]) for i in range(1, n)]#w_i = Q^i * v
    K = np.array(K)"""
    print(f'K = {K}')

    for index in range(1, n):
        K[index] = Q(K[index-1])

    print(f'New K = {K}') 

    eigenvalue = None

    for iteration in range(max_iter):
        for i in range(0, n):
            K[i] = Q(K[i]) #compute w_i_new = Q * w_i

        K = gram_schmidt(K) #orthonormalise w
        print(K)
        eigenvalue_new = np.zeros(n)
        for i in range(0, n):
            eigenvalue_new[i] = np.vdot(K[i], Q(K[i])).real #computing eigenvalues

        #check for convergence
        if eigenvalue is not None and np.abs(np.max(eigenvalue_new - eigenvalue)) < tol:
            print(f'converged after {iteration} iterations.')
            return eigenvalue_new, K
        
        #update K and eigenvalue for next step
        K = K
        eigenvalue = eigenvalue_new

    # If maximum iterations are reached without convergence, raise an error
    raise RuntimeError(f'Arnoldi method failed to converge within {max_iter} iterations.')

"""b = np.array([[4, 1],
            [2, 5]])

def Q(x):
    return np.matmul(b,x) #np.dot(A,x)  

largest_eigenvalue, eigenvector = arnoldi_method4(Q, n=2, tol=1e-3, max_iter=10000) 

print("Largest eigenvalue:", largest_eigenvalue)
print("Corresponding eigenvector:", eigenvector)

lowest_eigenvalue = 1/largest_eigenvalue

print("lowest eigenvalue:", lowest_eigenvalue)"""
