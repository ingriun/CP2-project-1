##testing linearity
import numpy as np
import numpy.random as random
from subproject1 import hamiltonian, derivative, kineticEnergy, strang_splitting_integrator, second_order_integrator, ndim_Ones
from subproject1 import N, mu, epsilon, tau_hat, dim

def ndim_Random(dim, N):


    list = [N for x in range(0,dim)]
    
    # tuple containing the shape of the lattice
    tuplet = tuple(list)
        
    #array = np.random.rand(*tuplet, dtype=complex)
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

    k = np.array([random.randint(-10, 10) for x in range(0,dim)])
    #print("k :", k)

    psi = ndim_Ones(dim, N)
    #print(psi)

    for index in np.ndindex(psi.shape):
        psi[index] = np.exp(2 * np.pi * 1j * np.vdot(index,k) / N)
    
    #print("Psi :")
    #print(psi)

    eigenvalue = (2/(mu*epsilon**2)) * (np.sin(np.pi * k/N))**2
    #print("Eigenvalue : ", eigenvalue)

    rightSide = np.sum(eigenvalue) * psi
    #print("rightSide : ", rightSide)

    leftSide = noPotentialHamiltonian(psi)
    #print("leftSide : ", leftSide)

    difference = leftSide - rightSide

    differenceMax = np.max(difference)
    #print("absolute difference : ")
    return np.abs(differenceMax) #np.allclose(leftSide, rightSide)


def unitarityTest(dim, N):
    """

    """
    psi = ndim_Random(dim, N)

    initialNorm = np.linalg.norm(psi)

    transformedNorm = np.linalg.norm(strang_splitting_integrator(psi, tau_hat))

    return np.allclose(initialNorm, transformedNorm)


def testIntegrators(dim, N, tau_hat):
    
    psi = ndim_Random(dim, N)
    i = 0
    while i < 50:
        tau_hat = tau_hat/10

        rightSide = second_order_integrator(psi, tau_hat)

        leftSide = strang_splitting_integrator(psi, tau_hat)

        divergence = np.abs(leftSide - rightSide)

        div_max = np.linalg.norm(divergence)

        i = i + 1

        print(div_max)
    tau_hat = 1


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
    print("eigenvalueTest :")
    while i < 10:
        print(eigenvalueTest(dim, N))
        i = i+1

def testUnitarity(dim, N):
    i=0
    boo = True
    while i < 50 and boo == True:
        boo = unitarityTest(dim, N)
        i = i+1
    print("unitarityTest :")
    print(i)
    print(boo)


"""test1 = testLinearity(dim, N)
test2 = testHermiticity(dim, N)
test3 = testPositivity(dim, N)
test4 = testEigenvalue(dim, N)
test5 = testUnitarity(dim, N)
test6 = testIntegrators(dim, N, tau_hat)
"""
test = testEigenvalue(dim, N)