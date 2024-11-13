##testing linearity
import numpy as np
import numpy.random as random
from subproject1 import hamiltonian, derivative, kineticEnergy, strang_splitting_integrator, second_order_integrator
from math import pi

#######initializing variables######
epsilon = 0.1
mu = 0.02
dim = 2
N = 4
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

    k = [random.randint(-50, 50) for x in range(0,dim)]
    print("k :")
    print(k)


    k_prime = k[random.randint(0,dim)]

    psi = np.exp(2* pi * 1j * n * k_prime / N)
    print("Psi : ", psi)

    eigenvalue = (2/(mu*epsilon**2)) * np.sin(pi*k_prime/N)**2
    print("Eigenvalue : ", eigenvalue)

    rightSide = eigenvalue * psi
    print("rightSide : ", rightSide)

    leftSide = noPotentialHamiltonian(psi)
    print("leftSide : ", leftSide)

    return np.allclose(leftSide, rightSide)


def unitarityTest(dim, N):
    """

    """
    psi = ndim_Random(dim, N)

    initialNorm = np.linalg.norm(psi)

    transformedNorm = np.linalg.norm(strang_splitting_integrator(psi))

    return np.allclose(initialNorm, transformedNorm)


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

def testUnitarity(dim, N):
    i=0
    boo = True
    while i < 50 and boo == True:
        boo = unitarityTest(dim, N)
        i = i+1
    print("unitarityTest :")
    print(i)
    print(boo)


def testIntegrators(dim, N):
    global tau_hat

    psi = ndim_Random(dim, N)

    i = 0
    boo = True
    while i < 15 and boo == True:
        tau_hat = tau_hat/10

        rightSide = second_order_integrator(psi)

        leftSide = strang_splitting_integrator(psi)

        i = i + 1
        boo = np.allclose(rightSide, leftSide)

    print("testIntegrators")
    print(i)
    print(boo)


test1 = testLinearity(dim, N)
test2 = testHermiticity(dim, N)
test3 = testPositivity(dim, N)
test4 = testEigenvalue(dim, N)
test5 = testUnitarity(dim, N)
test6 = testIntegrators(dim, N)
"""

x = eigenvalueTest(dim,N)
"""
