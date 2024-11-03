import numpy as np


#######initializing variables######
epsilon = 1
mu = 1
dim = 1
N = 6 
tau_hat = 0.1
###########


# n-dimensional array
def ndimTensor(dim, N):
    

    list = [N for x in range(0,dim)]
    
    # tuple containing the shape of the array
    tuplet = tuple(list)
        
    array = np.ndarray(tuplet)
    return array

# Initialize Psi
Psi = ndimTensor(dim, N)



def hamiltonian(Psi, V, x):

    dim = np.ndim(Psi)
    # Initialise the hamiltonian as 2dim-array  
    H_hat = np.shape(Psi)
    
    # calculate the lattice discretized 2nd derivative of Psi
    Psi2nd_array = np.zeros(dim)
    
    
    Psi2nd_float = np.sum(Psi2nd_array)
    
    H_hat = -1/(2*mu*epsilon**2)*Psi2nd_float + V*Psi
    
    return H_hat

x = np.linspace(0, 5, N)
Psi[dim] = np.ones(N)
Pot = m

print(hamiltonian(Psi, dim))

# Euler Integrator
def euler_integrator(Psi, H_hat):
    return Psi - 1j * tau_hat * H_hat @ Psi

# Second-order Integrator
def second_order_integrator(Psi, H_hat):
    return Psi - 1j * tau_hat * H_hat @ Psi - (tau_hat**2 / 2) * (H_hat @ H_hat @ Psi)

def strang_splitting_integrator(Psi, H_hat):
    # Split Hamiltonian into kinetic and potential parts
    V_half = np.exp(-1j * (tau_hat / 2) * V)  # e^(-i*tau_hat/2 * V)
    
    # Apply potential
    eta = V_half * Psi

    #fourier transform to momentum space
    eta_tilde = np.fft.fftn(eta)

    K_hat = hamiltonian(eta, np.zeros_like(V))
    K_exp = np.exp(-1j * (tau_hat) * K_hat)

    #apply kinetic part
    xi = np.fft.ifft2(K_exp * eta_tilde) #transform back to position space

    return V_half * xi