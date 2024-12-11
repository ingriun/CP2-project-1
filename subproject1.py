import numpy as np
import numpy.random as random
from math import pi
import matplotlib.pyplot as plt 

#######initializing variables######
N = 201
epsilon = 0.03
mu = 100
dim = 2
tau_hat = 0.01
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
    array = np.random.random(tuplet).astype(complex)
    return array



################# hamiltonian function ##################


def laplacian(psi):
    # Boundary conditions inherent to np.roll
    psi_2nd = np.zeros_like(psi)
    
    for axis in range(psi.ndim):
        psi_2nd += np.roll(psi, -1, axis=axis) - 2 * psi + np.roll(psi, 1, axis=axis) 

    return psi_2nd

def kineticEnergy(psi):
    k_hat = (-1/(2*mu*epsilon**2))*(laplacian(psi))
    return k_hat

def potential(psi):    
    N = psi.shape[0]

    # Potential centered in 0 to obtain the double-well property
    coordinates_centered = [np.linspace(-N//2+1, N//2 , N) for dim in psi.shape]

    grids = np.meshgrid(*coordinates_centered, indexing="ij")
    
    # Compute radial distance from the center
    r = np.sqrt(sum(g**2 for g in grids))
    
    v_hat = mu / 8 * ((epsilon**2 * r**2 - 1)**2)

    return v_hat


def hamiltonian(psi):

    v_hat = potential(psi)

    k_hat = kineticEnergy(psi)

    h_hat = k_hat + v_hat * psi
    
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
    pot = potential(psi)
    v_half = np.exp(-1j * (tau_hat / 2) * pot)
    
    # Apply potential
    eta = v_half * psi

    #fourier transform to momentum space
    eta_tilde = np.fft.fftn(eta)

    #print(psi.shape)
    # Define the Fourier space wave numbers
    k = np.fft.fftfreq(psi.shape[0], d=1) * 2 * np.pi  # FFT frequencies, scaled by 2π
    #print(k)
    k_mesh = np.meshgrid(*([k] * psi.ndim), indexing='ij')  # Create a meshgrid for each dimension

    # Calculate eigenvalues of k_hat in Fourier space using the known formula
    k_eigenvalues = sum((2 / (mu * epsilon**2)) * np.sin(k_dim / 2)**2 for k_dim in k_mesh)

    # Define the kinetic evolution operator in Fourier space
    def kinetic_evolution_operator(tau_hat):
        return np.exp(-1j * tau_hat * k_eigenvalues)

    k_exp = kinetic_evolution_operator(tau_hat)

    #apply kinetic part
    xi = np.fft.ifftn(k_exp * eta_tilde) #transform back to position space

    return v_half * xi

strang_splitting_integrator(ndim_Random(dim, 5), tau_hat)