import numpy as np

#######initializing variables######
h_bar = 1.05 * 10**(-31)
m = float
dim = int
Psi = np.zeros(dim, dim)
V = 'formula'
a = 'lattice spacing'
###########


def hamiltonian(Psi, dim, V, x):
    # Initialise the hamiltonian as 2dim-array  
    H = [][]

    # calculate the lattice discretized 2nd derivative of Psi
    Psi2nd_array = []

    for n in (0..dim):
        Psi2nd_array[n] = (Psi[n][x+a] - 2**Psi[n] + Psi[n][x-a])/a

    Psi2nd_float = np.sum(Psi2nd_array)
    H = -(h_bar**2)/(2*m)*Psi2nd + V*Psi

    return Hpsi