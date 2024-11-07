import numpy as np


#######initializing variables######
epsilon = 1
mu = 1
dim = 1
N = 6 
tau_hat = 0.1
###########


# n-dimensional array of ones for psi in lattice
def ndim_Ones(dim, N):


    list = [N for x in range(0,dim)]
    
    # tuple containing the shape of the lattice
    tuplet = tuple(list)
        
    array = np.ones(tuplet)
    return array


# Initialize psi
psi = ndim_Ones(dim, N)


def derivative(psi):
    # Initialize psi_2nd to have the same shape as psi
    psi_2nd = np.zeros(psi.shape)  

    psi_2nd = np.roll(psi, -1) - 2*psi + np.roll(psi, 1) # Calculate psi_2nd
    
    return psi_2nd

y = derivative(psi)
print("2nd-derivative :")
print(y)

def kineticEnergy(psi):
    k_hat = np.ones(psi.shape)
    k_hat = -1/(2*mu*epsilon**2)*k_hat


def potential(psi):
    # Initialize V_psi to have the same shape as psi
    v_hat = np.zeros(psi.shape) 

    # Number of point along an axis
    N = list(psi.shape)[0]

    # Create an array of N-indices in 1 dimension
    indices = np.arange(N)

    # Calculate the potential
    v_hat = mu/8*((epsilon**2 * indices**2 - 1)**2)

    return v

v = potential(psi)
print("Potential : ")
print(v)


def hamiltonian(psi):

    psi_2nd = derivative(psi)    

    v_hat = potential(psi)

    k_hat = kineticEnergy(psi)
    # Calculate the hamiltonian
    h_hat = k_hat*psi_2nd + v_hat * psi
    
    return h_hat


print(hamiltonian(psi))

# Euler Integrator
def euler_integrator(psi): # add tau_hat in parameters?
    return psi - 1j * tau_hat * hamiltonian(psi)

# Second-order Integrator
def second_order_integrator(psi):   # add tau_hat in parameters?
    return psi - 1j * tau_hat * hamiltonian(psi) - (tau_hat**2 / 2) * (hamiltonian(hamiltonian(psi)))

def strang_splitting_integrator(psi):
    # Split Hamiltonian into kinetic and potential parts
    v_half = np.exp(-1j * (tau_hat / 2) * potential(psi))  # e^(-i*tau_hat/2 * V)
    
    # Apply potential
    eta = v_half * psi

    #fourier transform to momentum space
    eta_tilde = np.fft.fftn(eta)

    v_eta = potential(eta) #calculate potential
    k_hat = hamiltonian(eta) - v_eta #calculate kinetic hamiltonian by removing the V part (setting V=0)
    k_exp = np.exp(-1j * (tau_hat) * k_hat) #kinetic evolution oeprator

    #apply kinetic part
    xi = np.fft.ifft2(k_exp * eta_tilde) #transform back to position space

    return v_half * xi