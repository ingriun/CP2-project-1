import numpy as np
import numpy.random as random
from math import pi
import matplotlib.pyplot as plt 

#######initializing variables######
N = 99
epsilon = 0.03*101/N
mu = 0.2
dim = 1
tau_hat = 1
###########


# n-dimensional array of ones for psi in lattice
def ndim_Ones(dim, N):


    list = [N for x in range(0,dim)]
    
    # tuple containing the shape of the lattice
    tuplet = tuple(list)
        
    array = np.ones(tuplet, dtype=complex)
    return array

def ndim_Random(dim, N):


    list = [N for x in range(0,dim)]
    
    # tuple containing the shape of the lattice
    tuplet = tuple(list)
        
    array = random.rand(*tuplet)
    return array

# Initialize psi
psi = ndim_Random(dim, N)


def derivative(psi):
    # Initialize psi_2nd to have the same shape as psi 
    psi_2nd = np.roll(psi, -1) - 2*psi + np.roll(psi, 1) # Calculate psi_2nd
    return psi_2nd


def kineticEnergy(psi):
    array = np.ones(psi.shape)
    k_hat = -1/(2*mu*epsilon**2)*array

    return k_hat



def potential(psi):

    ones = np.ones(psi.shape)
    
    N = psi.shape[0]

    coordinates_centered = np.linspace(N//2, -N//2 + 1, N)

    v_hat = coordinates_centered * ones

    for index in np.ndindex(psi.shape):
        v_hat[index] = mu/8*((epsilon**2 * v_hat[index]**2 - 1)**2)

    """a = psi.shape[0]
    N = np.arange(a)
    fig, ax = plt.subplots()
    ax.plot(N,v_hat)
    plt.show()"""

    return v_hat

v = potential(psi)
print(v)



def hamiltonian(psi):

    psi_2nd = derivative(psi)    

    v_hat = potential(psi)

    k_hat = kineticEnergy(psi)
    # Calculate the hamiltonian
    h_hat = k_hat*psi_2nd + v_hat * psi
    
    return h_hat

h = hamiltonian(psi)

# Euler Integrator
def euler_integrator(psi): # add tau_hat in parameters?
    return psi - 1j * tau_hat * hamiltonian(psi)

# Second-order Integrator
def second_order_integrator(psi, tau_hat):   # add tau_hat in parameters?
    return psi - 1j * tau_hat * hamiltonian(psi) - (tau_hat**2 / 2) * (hamiltonian(hamiltonian(psi)))

def strang_splitting_integrator(psi, tau_hat):
    # Split Hamiltonian into kinetic and potential parts
    v_half = np.exp(-1j * (tau_hat / 2) * potential(psi))  # e^(-i*tau_hat/2 * V)
    
    # Apply potential
    eta = v_half * psi

    #fourier transform to momentum space
    eta_tilde = np.fft.fftn(eta)


    # Define the Fourier space wave numbers
    k = np.fft.fftfreq(N, d=epsilon) * 2 * np.pi  # FFT frequencies, scaled
    k_mesh = np.meshgrid(*([k] * dim), indexing='ij')  # Create a meshgrid for each dimension

    # Calculate eigenvalues of K_hat in Fourier space using the known formula
    k_eigenvalues = sum((2 / (mu * epsilon**2)) * np.sin(k_dim / 2)**2 for k_dim in k_mesh)

    # Define the kinetic evolution operator in Fourier space
    def kinetic_evolution_operator(tau_hat):
        return np.exp(-1j * tau_hat * k_eigenvalues)

    k_exp = kinetic_evolution_operator(tau_hat)

    #apply kinetic part
    xi = np.fft.ifftn(k_exp * eta_tilde) #transform back to position space

    # or xi = np.fft.ifftn(K_exp * eta_tilde)
    return v_half * xi


#bla = strang_splitting_integrator(psi, tau_hat)