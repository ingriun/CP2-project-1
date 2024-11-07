##testing linearity
import numpy as np
import numpy.random as random
from subproject1 import hamiltonian

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



def eigenvalueTest(psi):
    """
    input parameters:

    output parameters:

    """
    leftSide = hamiltonian(psi)

    return
