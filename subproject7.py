import numpy as np
from subproject1 import hamiltonian, ndim_Random, dim, N, ndim_Ones


def conjugateGradient(mat, tol, max_iter):
    b = 1
    x = 0*ndim_Ones(mat.ndim, mat.shape[0])
    r_new = ndim_Ones(mat.ndim, mat.shape[0]) * b - np.dot(mat,x)
    r_old = r_new
    p = r_new

    k=0
    while k < max_iter:
        mat_p = np.dot(mat,p)
        alpha = (np.dot(np.transpose(r_new),r_new))/(np.dot(np.transpose(p),mat_p))
        x = x + np.dot(alpha,p)
        r_new = r_new - np.dot(alpha,mat_p)

        if r_new < tol:
            mat_inv = x
            break

        beta = (np.dot(np.transpose(r_new),r_new))/(np.dot(np.transpose(r_old),r_old))
        p = r_new + beta*p
        r_old = r_new
        k = k+1
    return mat_inv

def power_method(Q, tol=1e-6, max_iter=10000):
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
        eigenvalue_new = np.vdot(w, Q(w)).real #using Hermiticity of Q

        #check for convergence
        if eigenvalue is not None and np.abs(eigenvalue_new - eigenvalue) < tol:
            print(f"converged after {iteration} iterations.")
            return eigenvalue_new, w
        
        eigenvalue = eigenvalue_new
        v=w #update w for next iteration

    print("Maximum iteration reached without convergence.")
    return eigenvalue, w

#defire Hamiltonian as the function for the operator
#def apply_hamiltonian(psi):
    #return hamiltonian(psi)

def gram_schmidt(V):
    """Define the gram-schmidt process to orthonormalise the eigenvectors

    Parameters:
    V: ndarray
        Consisting of column vectors that needs to be orthonormalised.

    Returns:
    U: ndarray
        Orthonormalised column vectors    
    """
    n, k = V.shape()
    U = np.zeros(n, k)
    U[:, 0] = V[:, 0]/np.linalg.norm(V[:, 0])

    for i in range(1, k):
        U[:, i] = V[:, i]

        for j in range(i):
            U[:, i] = U[:, i] - np.dot(U[:, j], U[:, i]) * U[:, j]

        U[:, i] = V[:, i]/np.linalg.norm(V[:, i])

    return U

def arnoldi_method():
    return

#run the power method
largest_eigenvalue, eigenvector = power_method(hamiltonian)

print("Largest eigenvalue:", largest_eigenvalue)
print("Corresponding eigenvector:", eigenvector)


