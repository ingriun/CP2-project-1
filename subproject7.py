import numpy as np
from subproject1 import hamiltonian, ndim_Random, dim, N, ndim_Ones
import numpy.random as random
import matplotlib.pyplot as plt


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
    v = ndim_Random(1,N) #choosing a random v
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


largest_eigenvalue, eigenvector = arnoldi_method4(hinv, n=4, N=56160, tol=1e-6,max_iter=1000)

lowest_eigenvalue = 1/largest_eigenvalue

print("Lowest eigenvalue (N=351, ε=0.1):", lowest_eigenvalue)
#print("Corresponding eigenvector:", eigenvector)


##################### Plots for L -> infinity and a ->0 ##############################

# Input data
"""N_values = [3, 15, 51, 101, 151, 201, 251, 301, 351, 401, 501, 701, 1001, 1501, 2001]
eigenvalue_arrays = [
    [2.49966668, 752.49973762, 752.49959393, 2.49966668],  # array for N=3
    [2.49068075, 45.71696764, 45.71896389, 167.92481792],  # array for N=15
    [2.39155668, 6.19741187, 6.17105531, 17.49632544],  # array for N=51
    [1.99584354, 3.11225907, 3.10947104, 5.97252104], #array for N=101
    [1.16158291, 2.0812058,  2.44738283, 3.50289752], #array for N=151
    [0.42118223, 1.19449557, 1.82408095, 2.4067554 ], #array for N=201
    [0.33183823, 0.61383311, 1.25465675, 1.68996566], #array for N=251
    [0.48369551, 0.4889431,  1.38047371, 1.41166015], #array for N=301
    [0.48662402, 0.48664083, 1.3976937,  1.39936959], #array for N=351
    [0.48662492, 0.48664001, 1.39772967, 1.39933445], #array for N=401
    [0.48662492, 0.48664001, 1.39771152, 1.39935271], #array for N=501
    [0.48662492, 0.48664001, 1.39770837, 1.39935571], #array for N=701
    [0.48662502, 0.48663991, 1.39770358, 1.39936054], #array for N=1001
    [0.48662492, 0.48664001, 1.39772295, 1.39934139], #array for N=1501
    [0.48662492, 0.48664001, 1.39770216, 1.3993621 ], #array for N=2001
]

# Transpose data to get 4 lists of eigenvalues
eigenvalues_transposed = list(zip(*eigenvalue_arrays))

# Plotting
plt.figure(figsize=(10, 6))
for i, eigenvalues in enumerate(eigenvalues_transposed):
    plt.plot(N_values[:len(eigenvalues)], eigenvalues, label=f'Eigenvalue {i+1}')

plt.xlabel('N')
plt.ylabel('Eigenvalue')
plt.ylim(0,40)
plt.xlim(0,2000)
plt.title('Eigenvalues as L approaches infinity')
plt.legend()
plt.grid(True)
plt.show()

#input data
epsilon_values = [0.1, 0.01, 0.005, 0.0025, 0.001, 0.00125, 0.000625]
eigval_arrays = [
    [0.48084574, 0.48086378, 1.37361197, 1.37418283], #ε=0.1
    [0.48662495, 0.48663998, 1.39770502, 1.39935906], #ε=0.01
    [0.4866681,  0.48668317, 1.39797614, 1.39945067], #ε=0.005
    [0.48667889, 0.48669396, 1.39792743, 1.39959036], #ε=0.0025
    [0.48668192, 0.48669698, 1.39795031, 1.39959302], #ε=0.001
    [0.48668159, 0.48669666, 1.39794025, 1.39960033], #ε=0.00125
    [0.48668227, 0.48669733, 1.39794344, 1.39960261], #ε=0.000625
]

# Transpose data to get 4 lists of eigenvalues
eigenvalues_transposed = list(zip(*eigval_arrays))

# Plotting
plt.figure(figsize=(10, 6))
for i, eigenvalues in enumerate(eigenvalues_transposed):
    plt.plot(epsilon_values[:len(eigenvalues)], eigenvalues, label=f'Eigenvalue {i+1}')

plt.xlabel('ε')
plt.ylabel('Eigenvalue')
plt.ylim(0,5)
plt.xlim(0.000625,0.1)
plt.title('Eigenvalues as a approaches 0')
plt.legend()
plt.grid(True)
plt.show()"""



# Extract the first eigenvector 
eigenfunction = eigenvector[0].real  

# Define x-axis values 
x_values = np.arange(len(eigenfunction))  

# Plot the eigenfunction
plt.figure(figsize=(10, 6))
plt.plot(x_values, eigenfunction, label="Eigenfunction 1")

"""for i, eigenfunction in enumerate(eigenvector):
    plt.plot(x_values, eigenfunction, label=f"Eigenfunction {i+1}")"""


plt.xlabel("Lattice Site Index")
plt.xlim(20000,35000)
plt.ylabel("Amplitude")
plt.title("Eigenfunction of the Hamiltonian (ε=0.000625)")
plt.grid(True)
plt.legend()
plt.show()

"""fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()  # Flatten to easily index subplots

# Plot each eigenvector in a separate subplot
for i, ax in enumerate(axes):
    ax.plot(x_values, eigenvector[i], label=f"Eigenfunction {i+1}", marker='o')
    ax.set_xlabel("Position / Index")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Eigenfunction {i+1}")
    ax.grid(True)
    ax.legend()

# Adjust layout
plt.tight_layout()
plt.show()"""

