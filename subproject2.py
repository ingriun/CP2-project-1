##testing linearity
import numpy as np
import numpy.random as random

def linearityTest(H, psi1, psi2):
    """
    testing linearity of the hamiltonian operator

    input parameters:
    - H: Hamiltonian operator, function
    - psi1, psi2: sample wavefunctions

    output parameters:
    - True if linear, False if not

    """
    a = random.randint(0, 50)
    b = random.randint(0, 50)

    psiLeft = a * psi1 + b * psi2

    leftSide = H(psiLeft)

    rightSide = H(a*psi1) + H(b*psi2)

    return np.allclose(leftSide, rightSide)



"""
def test_linearity(H, psi1, psi2, a, b):

Test the linearity of the Hamiltonian operator.

Parameters:
- H: Hamiltonian operator.
- psi1, psi2: Two sample wavefunctions.
- a, b: Scalars for the linearity test.

Returns:
- True if linearity holds, False otherwise.

# Compute H(a * psi1 + b * psi2)
left_side = H @ (a * psi1 + b * psi2)

# Compute a * H(psi1) + b * H(psi2)
right_side = a * (H @ psi1) + b * (H @ psi2)

# Check if the two sides are approximately equal
return np.allclose(left_side, right_side)

"""

def hermiticityTest(H, psi1, psi2):
    """
    testing if the hamiltonian operator is hermitian.

    input parameters:
    - H: Hamiltonian operator, function
    - psi1, psi2: sample wavefunctions

    output parameters:
    - True if hermitian, false if not

    """

    H_psi1 = H(psi1)

    H_psi2 = H(psi2)

    leftSide = np.sum((np.conj(psi1) * H_psi1))

    rightSide = np.sum((np.conj(H_psi1) * psi2))

    return np.allclose(leftSide, rightSide)
"""

def test_hermitian(H, psi1, psi2):

Test the Hermitian property of the Hamiltonian operator.

Parameters:
- H: Hamiltonian operator.
- psi1, psi2: Two sample wavefunctions.

Returns:
- True if the Hermitian property holds, False otherwise.

# Inner products
left_side = np.vdot(psi1, H @ psi2)  # <psi1 | H psi2>
right_side = np.vdot(H @ psi1, psi2)  # <H psi1 | psi2>

# Check if left_side equals the complex conjugate of right_side
return np.allclose(left_side, right_side)

"""

def positivityTest():
    """
    testing if the hamiltonian is positive when the potential is positive

    input parameters:

    output parameters:

    """

    return
