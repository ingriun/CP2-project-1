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


    eigenvalue = np.sum((2/(mu*epsilon**2)) * (np.sin(np.pi * k/N))**2)

    rightSide = eigenvalue*psi

    leftSide = kineticEnergy(psi)

    difference = leftSide - rightSide

    differenceMax = np.max(difference)

    return np.abs(differenceMax) 


def second_orderLinearity(dim, N, tau_hat):
    """
    Testing linearity of the second order integrator

    Wave functions are random in each iteration

    Output:
    - True if linear, False if not

    """
    a = random.uniform(-50, 50) + 1j * random.uniform(-50, 50)
    b = random.uniform(-50, 50) + 1j * random.uniform(-50, 50)

    psi1 = ndim_Random(dim, N)
    psi2 = ndim_Random(dim, N)

    psiLeft = a * psi1 + b * psi2

    leftSide = second_order_integrator(psiLeft, tau_hat)

    rightSide = second_order_integrator(a*psi1, tau_hat) + second_order_integrator(b*psi2, tau_hat)

    return np.allclose(leftSide, rightSide)


def strang_splittingLinearity(dim, N, tau_hat):
    """
    Testing linearity of the strang-splitting integrator

    Wave functions are random in each iteration

    Output:
    - True if linear, False if not

    """
    a = random.uniform(-50, 50) + 1j * random.uniform(-50, 50)
    b = random.uniform(-50, 50) + 1j * random.uniform(-50, 50)

    psi1 = ndim_Random(dim, N)
    psi2 = ndim_Random(dim, N)

    psiLeft = a * psi1 + b * psi2

    leftSide = strang_splitting_integrator(psiLeft, tau_hat)

    rightSide = strang_splitting_integrator(a*psi1, tau_hat) + strang_splitting_integrator(b*psi2, tau_hat)

    return np.allclose(leftSide, rightSide)


def unitarityTest(dim, N, tau_hat):
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

def testLinearity():
    print("---LinearityTest--- \n")

    print("For different number of dimension :")
    for d in range(1,5):
        i = 0 
        boo = True
        while i < 50 and boo == True:
            boo = linearityTest(d, 5)
            i = i+1
        print("dim = ",d," : ", boo)  
    
    print("For different value of N")
    for n in [15, 55, 91]:
        i = 0 
        boo = True
        while i < 50 and boo == True:
            boo = linearityTest(2, n)
            i = i+1
        print("N = ",n," : ", boo) 


def testHermiticity():
    print("---hermiticityTest--- \n")

    print("For different number of dimension :")
    for d in range(1,5):
        i=0
        boo = True
        while i < 50 and boo == True:
            boo = hermiticityTest(d, 5)
            i = i+1
        print("dim = ",d," : ", boo)  



def testPositivity():
    print("---positivityTest--- \n")

    print("For different number of dimension :")
    for d in range(1,5):
        i=0
        boo = True
        while i < 50 and boo == True:
            boo = positivityTest(d, 5)
            i=i+1
        print("dim = ",d," : ", boo)  

    print("For different value of N")
    for n in [15, 55, 91]:
        i = 0 
        boo = True
        while i < 50 and boo == True:
            boo = positivityTest(2, n)
            i = i+1
        print("N = ",n," : ", boo) 


# On Second order & Strang-Splitting integrators

def testLinearityIntegrators():
    print("---LinearityTest for Second-Order integrator--- \n")

    print("For different number of dimension :")
    for d in range(1,5):
        i = 0 
        boo = True
        while i < 50 and boo == True:
            boo = second_orderLinearity(d, 5, tau_hat=1.3)
            i = i+1
        print("dim = ",d," : ", boo) 
    
    print("For different value of N")
    for n in [15, 55, 91]:
        i = 0 
        boo = True
        while i < 50 and boo == True:
            boo = second_orderLinearity(2, n, tau_hat=1.3)
            i = i+1
        print("N = ",n," : ", boo)


    print("\n ---LinearityTest for Strang-Splitting integrator--- \n")

    print("For different number of dimension :")
    for d in range(1,5):
        i = 0 
        boo = True
        while i < 50 and boo == True:
            boo = strang_splittingLinearity(d, 5, tau_hat=1.3)
            i = i+1
        print("dim = ",d," : ", boo)

    print("\n For different value of N")
    for n in [15, 55, 91]:
        i = 0 
        boo = True
        while i < 50 and boo == True:
            boo = strang_splittingLinearity(2, n, tau_hat=1.3)
            i = i+1
        print("N = ",n," : ", boo)


# On Strang-Splitting integrator 

def testUnitarity():
    print("---unitarityTest--- \n")

    print("For different number of dimension :")
    for d in range(1,5):
        i=0
        boo = True
        while i < 50 and boo == True:
            boo = unitarityTest(d, 5, tau_hat=1.3)
            i = i+1
        print("dim = ",d," : ", boo)  

    print("For different value of N")
    for n in [15, 55, 91]:
        i = 0 
        boo = True
        while i < 50 and boo == True:
            boo = unitarityTest(2, n, tau_hat)
            i = i+1
        print("N = ",n," : ", boo) 



################### Absolute divergence tests ##################

# On eigenvalue

def testEigenvalue():
    print("---eigenvalueTest--- \n")

    print("For different number of dimension :\n")
    for d in range(1,5):
        print("dim = ", d, " :")
        i=0
        while i < 10:
            print(eigenvalueTest(d, N))
            i = i+1
        print("\n")
    
    print("\n")

    print("For different value of N : \n")
    for n in [15, 55, 91]:
        print("N = ",n," : ") 
        i = 0 
        while i < 10:
            print(eigenvalueTest(2, n))
            i = i+1
        print("\n")

# On Second order & Strang-Splitting integrators

def integDiv(psi,tau_hat):
    for i in range(4):
        print("tau_hat = ", tau_hat, " :")
        print("---> Divergence : ")

        rightSide = psi
        leftSide = psi

        for m in np.arange(0,1,tau_hat):
            rightSide = second_order_integrator(rightSide, tau_hat)
            leftSide = strang_splitting_integrator(leftSide, tau_hat)
                          
        divergence = np.abs(leftSide - rightSide)
        div_max = np.max(divergence)        

        print("-----------> ", div_max)
        tau_hat = tau_hat/10
    print("\n")


def testIntegrators():
    print("---integratorsTest--- \n")    

    print("For different number of dimension : \n")
    for d in range(1,5):
        print("dim = ",d," : ")

        psi = ndim_Random(d, 5)
        integDiv(psi, tau_hat=1)


    print("For different value of N \n")
    for n in [15, 55, 91]:
        print("N = ",n," : ") 

        psi = ndim_Random(1, n)
        integDiv(psi, tau_hat=1)




#################### Tests call ####################

print("\n ################################ \n")
test1 = testLinearity()
print("\n ################################ \n")
test2 = testHermiticity()
print("\n ################################ \n")
test3 = testPositivity()
print("\n ################################ \n")
test4 = testEigenvalue()
print("\n ################################ \n")
test5 = testUnitarity()
print("\n ################################ \n")
test6 = testLinearityIntegrators()
print("\n ################################ \n")
test7 = testIntegrators()
