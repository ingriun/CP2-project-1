import numpy as np
import numpy.random as random
from math import pi

#######initializing variables######
epsilon = 1
mu = 1
dim = 1
N = 100
tau_hat = 0.01
###########


# n-dimensional array of ones for psi in lattice
def ndim_Ones(dim, N):


    list = [N for x in range(0,dim)]
    
    # tuple containing the shape of the lattice
    tuplet = tuple(list)
        
    array = np.ones(tuplet)
    return array


# Initialize psi
#psi = ndim_Ones(dim, N)


def derivative(psi):
    # Initialize psi_2nd to have the same shape as psi 
    psi_2nd = np.roll(psi, -1) - 2*psi + np.roll(psi, 1) # Calculate psi_2nd
    print(psi_2nd)
    return psi_2nd

"""y = derivative(psi)
print("2nd-derivative :")
print(y)"""

def kineticEnergy(psi):
    array = np.ones(psi.shape)
    k_hat = -1/(2*mu*epsilon**2)*array

    return k_hat


"""k = kineticEnergy(psi)
print("Kinetic Energy :")
print(k)"""

def potential(psi):
    # Initialize V_psi to have the same shape as psi
    v_hat = np.zeros(psi.shape) 

    # Number of point along an axis
    N = list(psi.shape)[0]

    # Create an array of N-indices in 1 dimension
    indices = np.arange(N)

    # Calculate the potential
    v_hat = mu/8*((epsilon**2 * indices**2 - 1)**2)

    return v_hat




def hamiltonian(psi):

    psi_2nd = derivative(psi)    

    v_hat = potential(psi)

    k_hat = kineticEnergy(psi)
    # Calculate the hamiltonian
    h_hat = k_hat*psi_2nd + v_hat * psi
    
    return h_hat

#h = hamiltonian(psi)

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