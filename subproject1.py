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


def Derivative(psi):
    # Initialize psi_2nd to have the same shape as psi
    psi_2nd = np.zeros(psi.shape)  

    psi_2nd = np.roll(psi, -1) - 2*psi + np.roll(psi, 1) # Calculate psi_2nd
    
    return psi_2nd

y = Derivative(psi)
print("2nd-derivative :")
print(y)


def Potential(psi):
    # Initialize v_psi to have the same shape as psi
    v_psi = np.zeros(psi.shape)  

    return v_psi


def hamiltonian(psi):

    psi_2nd = Derivative(psi)    

    
    h_hat = -1/(2*mu*epsilon**2)*psi_2nd
    
    return h_hat

x = np.linspace(0, 5, N)
psi[dim] = np.ones(N)
Pot = m

print(hamiltonian(psi, dim))

# Euler Integrator
def euler_integrator(psi, h_hat):
    return psi - 1j * tau_hat * h_hat @ psi

# Second-order Integrator
def second_order_integrator(psi, h_hat):
    return psi - 1j * tau_hat * h_hat @ psi - (tau_hat**2 / 2) * (h_hat @ h_hat @ psi)

def strang_splitting_integrator(psi, h_hat):
    # Split Hamiltonian into kinetic and potential parts
    v_half = np.exp(-1j * (tau_hat / 2) * V)  # e^(-i*tau_hat/2 * V)
    
    # Apply potential
    eta = v_half * psi

    #fourier transform to momentum space
    eta_tilde = np.fft.fftn(eta)

    k_hat = hamiltonian(eta, np.zeros_like(V)) #calculate kinetic hamiltonian by setting V=0
    k_exp = np.exp(-1j * (tau_hat) * k_hat) #kinetic evolution oeprator

    #apply kinetic part
    xi = np.fft.ifft2(k_exp * eta_tilde) #transform back to position space

    return v_half * xi