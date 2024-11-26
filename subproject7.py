import numpy as np
from subproject1 import hamiltonian, ndim_Random, dim, N

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
def apply_hamiltonian(psi):
    return hamiltonian(psi)

#run the power method
largest_eigenvalue, eigenvector = power_method(apply_hamiltonian)

print("Largest eigenvalue:", largest_eigenvalue)
print("Corresponding eigenvector:", eigenvector)


