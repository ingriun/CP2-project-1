##testing linearity
import numpy as np
import numpy.random as random
from subproject1 import hamiltonian, potential
from math import pi

#######initializing variables######
epsilon = 1
mu = 1
dim = 1
N = 6 
tau_hat = 0.1
###########


def linearityTest(psi1, psi2):
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

    psiLeft = a * psi1 + b * psi2

    leftSide = hamiltonian(psiLeft)

    rightSide = hamiltonian(a*psi1) + hamiltonian(b*psi2)

    return np.allclose(leftSide, rightSide)


def hermiticityTest(psi1, psi2):
    """
    testing if the hamiltonian operator is hermitian.

    input parameters:
    - H: hamiltonian operator, function
    - psi1, psi2: sample wavefunctions

    output parameters:
    - True if hermitian, false if not

    """

    h_psi1 = hamiltonian(psi1)

    h_psi2 = hamiltonian(psi2)

    leftSide = np.vdot((psi1, h_psi2))

    rightSide = np.sum((h_psi1, psi2))

    return np.allclose(leftSide, rightSide)


def positivityTest(psi):
    """
    testing if the hamiltonian is positive when the potential is positive

    input parameters:

    output parameters:

    """
    h_psi = hamiltonian(psi)

    expValue =  np.vdot((psi, h_psi))

    return expValue >= 0



def eigenvalueTest(dim):
    """
    input parameters:

    output parameters:

    """

    n = np.arange(N)

    k = np.array(random.randint(-15, 15) for x in range(0,dim))

    psi = np.exp(2* np.pi * 1j * np.vdot(n, k)/N)

    v = potential(psi)

    eigenvalue = 1/(2*mu*epsilon**2) * (4*(pi**2)*(k**2))/(N**2) + v

    rightSide = eigenvalue * psi

    leftSide = hamiltonian(psi)

    return np.allclose(leftSide, rightSide)

def testProperties():

    for i in range(0, 50):
        
