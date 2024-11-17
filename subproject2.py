import numpy as np
import numpy.random as random
from subproject1 import hamiltonian, laplacian, kineticEnergy, strang_splitting_integrator, second_order_integrator, ndim_Ones, ndim_Random
from subproject1 import N, mu, epsilon, tau_hat, dim


########################## test functions ###########################

def linearityTest(dim, N):
    """
    Testing linearity of the hamiltonian operator

    Wave functions are random in each iteration

    Output:
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
    Testing if the hamiltonian operator is hermitian

    Wave functions are random in each iteration

    Output:
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
    Testing if the hamiltonian is positive when the potential is positive

    - Wave functions is random in each iteration
    - 
        
    Output:
   - True if positive, false if not

    """

    psi = ndim_Random(dim, N)

    h_psi = hamiltonian(psi)

    expValue =  np.vdot(psi, h_psi)

    return expValue >= 0



def noPotentialHamiltonian(psi):
    """ hamiltonian without potential part,
        only used in eigenvalueTest"""

    psi_2nd = laplacian(psi) 

    k_hat = kineticEnergy(psi)

    h_hat = k_hat*psi_2nd 
    
    return h_hat



def eigenvalueTest(dim, N):
    """
    Test if H.psi = E.psi 

    Output: 
    - absolute divergence of the maximum value of each part

    Goal:
    - output should be close to 0                      

    """

    k = np.array([random.randint(-10, 10) for x in range(0,dim)])

    psi = ndim_Ones(dim, N)

    for index in np.ndindex(psi.shape):
        psi[index] = np.exp(2 * np.pi * 1j * np.vdot(index,k) / N)


    eigenvalue = (2/(mu*epsilon**2)) * (np.sin(np.pi * k/N))**2

    rightSide = np.sum(eigenvalue) * psi

    leftSide = noPotentialHamiltonian(psi)

    difference = leftSide - rightSide

    differenceMax = np.max(difference)

    return np.abs(differenceMax) 

def linearityIntegrators(dim, N, integrator):
    """
    Testing linearity of the integrators

    Wave functions are random in each iteration

    Output:
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


def unitarityTest(dim, N):
    """
    Test the unitarity of the Strang-Splitting integrator
    - norm of psi should be equal to norm of integrator applied to psi

    Output:
    - True if yes, false if not
    """
    psi = ndim_Random(dim, N)

    initialNorm = np.linalg.norm(psi)

    transformedNorm = np.linalg.norm(strang_splitting_integrator(psi, tau_hat))

    return np.allclose(initialNorm, transformedNorm)



######################## Pass/Fail tests ############################

# On Hamiltonian

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


# On Strang-Splitting integrator 

def testUnitarity(dim, N):
    i=0
    boo = True
    while i < 50 and boo == True:
        boo = unitarityTest(dim, N)
        i = i+1
    print("unitarityTest :")
    print(i)
    print(boo)



################### Absolute divergence tests ##################

# On eigenvalue

def testEigenvalue(dim, N):
    i=0
    print("eigenvalueTest :")
    while i < 10:
        print(eigenvalueTest(dim, N))
        i = i+1


# On Second order & Strang-Splitting integrators

def testLinearityIntegrators(dim, N, tau_hat):
    i = 0 
    boo = True
    while i < 50 and boo == True:
        boo = linearityIntegrators(dim, N, second_order_integrator)
        i = i+1
    print("LinearityTest for Second-Order integrator:")
    print(i)
    print(boo)  
    i = 0 
    boo = True
    while i < 50 and boo == True:
        boo = linearityIntegrators(dim, N, strang_splitting_integrator)
        i = i+1
    print("LinearityTest for Strang-Splitting integrator:")
    print(i)
    print(boo)  


def testIntegrators(dim, N, tau_hat):

    psi = ndim_Random(dim, N)
    i = 0
    print("integratorsTest : ")
    while i < 50:
        tau_hat = tau_hat/10

        rightSide = second_order_integrator(psi, tau_hat)

        leftSide = strang_splitting_integrator(psi, tau_hat)

        divergence = np.abs(leftSide - rightSide)

        div_max = np.linalg.norm(divergence)

        i = i + 1

        print(div_max)
    tau_hat = 1



#################### Tests call ####################

test1 = testLinearity(dim, N)
test2 = testHermiticity(dim, N)
test3 = testPositivity(dim, N)
test4 = testEigenvalue(dim, N)
test5 = testUnitarity(dim, N)
test6 = testIntegrators(dim, N, tau_hat)
test7 = testLinearityIntegrators(dim, N, tau_hat)