import numpy as np

#######initializing variables######
epsilon = 1
mu = 1
dim = 1
N = 6
Psi = np.zeros(N, N) #N^d 
###########


def hamiltonian(Psi, dim, V, x):

    dim = np.ndim(Psi)
    # Initialise the hamiltonian as 2dim-array  
    H = np.zeros(dim, dim)

    # calculate the lattice discretized 2nd derivative of Psi
    Psi2nd_array = np.zeros(dim)

    for n in (0,dim):
        Psi2nd_array[n] = (Psi[n][x[n]+a] - 2**Psi[n] + Psi[n][x[n]-a])/a

    Psi2nd_float = np.sum(Psi2nd_array)
    H = -(h_bar**2)/(2*m)*Psi2nd_float + V*Psi

    return H

x = np.linspace(0, 5, N)
Psi[dim] = np.ones(N)
Pot = m

print(hamiltonian(Psi, dim, ))