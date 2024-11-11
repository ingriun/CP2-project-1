##testing linearity
import numpy as np
import numpy.random as random
from subproject1 import hamiltonian, derivative, kineticEnergy
#from math import pi

#######initializing variables######
epsilon = 1
mu = 1
dim = 2
N = 3
tau_hat = 0.1
###########

def ndim_Random(dim, N):


    list = [N for x in range(0,dim)]
    
    # tuple containing the shape of the lattice
    tuplet = tuple(list)
        
    array = np.random.rand(*tuplet)
    return array


def linearityTest(dim, N):
    """
    testing linearity of the hamiltonian operator

    input parameters:
    - h: hamiltonian operator, function
    - psi1, psi2: sample wavefunctions

    output parameters:
    - True if linear, False if not

    """
    a = random.uniform(-50, 50) + 1j * random.uniform(-50, 50)
    b = random.uniform(-50, 50) + 1j * random.uniform(-50, 50)

    psi1 = ndim_Random(dim, N)
    psi2 = ndim_Random(dim, N)

    psiLeft = a * psi1 + b * psi2

    leftSide = hamiltonian(psiLeft)

    rightSide = hamiltonian(a*psi1) + hamiltonian(b*psi2)

    return np.allclose(leftSide, rightSide)


def hermiticityTest(dim, N):
    """
    testing if the hamiltonian operator is hermitian.

    input parameters:
    - H: hamiltonian operator, function
    - psi1, psi2: sample wavefunctions

    output parameters:
    - True if hermitian, false if not

    """
    psi1 = ndim_Random(dim, N)
    psi2 = ndim_Random(dim, N)

    h_psi1 = hamiltonian(psi1)

    h_psi2 = hamiltonian(psi2)

    leftSide = np.vdot(psi1, h_psi2)

    rightSide = np.vdot(h_psi1, psi2)

    return np.allclose(leftSide, rightSide)


def positivityTest(dim, N):
    """
    testing if the hamiltonian is positive when the potential is positive

    input parameters:

    output parameters:

    """

    psi = ndim_Random(dim, N)

    h_psi = hamiltonian(psi)

    expValue =  np.vdot(psi, h_psi)

    return expValue >= 0


def noPotentialHamiltonian(psi):

    psi_2nd = derivative(psi)    

    k_hat = kineticEnergy(psi)
    # Calculate the hamiltonian
    h_hat = k_hat*psi_2nd 
    
    return h_hat



def eigenvalueTest(dim, N):
    """
    input parameters: dim, N

    output parameters: True if H(psi_k) = E_k*psi_k for one given eigenvector
                       False if not                          

    """

    n = np.arange(N)

    k = [random.randint(-15, 15) for x in range(0,dim)]
    print("k :")
    print(k)

    # Choose 1 value of k 
    k_prime = k[random.randint(0,dim)]

    # wave function with the chosen value of k 
    psi = np.exp(2* np.pi * 1j * n*k_prime /N)

    psi_2nd  = derivative(psi)


    # Eigenvalue with the chosen value of k
    eigenvalue = 1/(2*mu*epsilon**2) * psi_2nd / psi # potential = 0

    rightSide = eigenvalue * psi

    leftSide = noPotentialHamiltonian(psi)

    return np.allclose(leftSide, rightSide)


def unitarityTest(psi, dim):
    """

    """
    

    return


def testLinearity(dim, N):
    i = 0 
    boo = True
    while i < 50 and boo == True:
        boo = linearityTest(dim, N)
        i = i+1
    print("LinearityTest :")
    print(i)
    print(boo)  


def testHermiticity(dim, N):
    i=0
    boo = True
    while i < 50 and boo == True:
        boo = hermiticityTest(dim, N)
        i = i+1
    print("hermiticityTest :")
    print(i)
    print(boo)


def testPositivity(dim, N):
    i=0
    boo = True
    while i < 50 and boo == True:
        boo = positivityTest(dim, N)
        i=i+1
    print("positivityTest :")
    print(i)
    print(boo)


def testEigenvalue(dim, N):
    i=0
    boo = True
    while i < 50 and boo == True:
        boo = eigenvalueTest(dim, N)
        i = i+1
    print("eigenvalueTest :")
    print(i)
    print(boo)
    return boo


test1 = testLinearity(dim, N)
test2 = testHermiticity(dim, N)
test3 = testPositivity(dim, N)

test4 = testEigenvalue(dim, N)