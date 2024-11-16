import numpy as np
import numpy.random as random
from math import pi
import matplotlib.pyplot as plt 

#######initializing variables######
N = 101
epsilon = 0.03*101 / N
mu = 1
dim = 1
tau_hat = 0.1
##################################

############### Initialize the lattice as array ###################


def ndim_Ones(dim, N):
    """Create an array of one in lattice dimensions (dim, N)"""

    list = [N for x in range(0,dim)]
    
    tuplet = tuple(list)
        
    array = np.ones(tuplet, dtype=complex)
    return array


def ndim_Random(dim, N):
    """Create an array of complex numbers in lattice dimensions (dim, N)"""

    list = [N for x in range(0,dim)]
    
    # tuple containing the shape of the lattice
    tuplet = tuple(list)
    
    # Create lattice array w/ complex numbers
    array = np.random.random(*tuplet).astype(complex)
    return array



################# hamiltonian function ##################


def derivative(psi):
    # Boundary conditions inherent to np.roll
    psi_2nd = np.roll(psi, -1) - 2*psi + np.roll(psi, 1) 
    return psi_2nd


def kineticEnergy(psi):
    array = np.ones(psi.shape)
    k_hat = -1/(2*mu*epsilon**2)*array
    return k_hat


def potential(psi):

    ones = np.ones(psi.shape)
    
    N = psi.shape[0]

    # Potential centered in 0 to obtain the double-well property
    coordinates_centered = np.linspace(N//2, -N//2 + 1, N)

    v_hat = coordinates_centered * ones

    for index in np.ndindex(psi.shape):
        v_hat[index] = mu/8*((epsilon**2 * v_hat[index]**2 - 1)**2)

    return v_hat


def hamiltonian(psi):

    psi_2nd = derivative(psi)    

    v_hat = potential(psi)

    k_hat = kineticEnergy(psi)

    h_hat = k_hat*psi_2nd + v_hat * psi
    
    return h_hat



#################### integrators ########################


# Euler Integrator
def euler_integrator(psi):
    return psi - 1j * tau_hat * hamiltonian(psi)

# Second-order Integrator
def second_order_integrator(psi, tau_hat):   
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

    # Calculate eigenvalues of k_hat in Fourier space using the known formula
    k_eigenvalues = sum((2 / (mu * epsilon**2)) * np.sin(k_dim / 2)**2 for k_dim in k_mesh)

    # Define the kinetic evolution operator in Fourier space
    def kinetic_evolution_operator(tau_hat):
        return np.exp(-1j * tau_hat * k_eigenvalues)

    k_exp = kinetic_evolution_operator(tau_hat)

    #apply kinetic part
    xi = np.fft.ifftn(k_exp * eta_tilde) #transform back to position space

    return v_half * xi