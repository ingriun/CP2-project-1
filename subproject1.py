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

# strang splitting integrator
def strang_splitting_integrator(Psi, H_hat):
    # Split Hamiltonian into kinetic and potential parts
    V_half = np.exp(-1j * (tau_hat / 2) * V)  # e^(-i*tau_hat/2 * V)
    
    eta = V_half * Psi #apply V_half to psi in position space

    #fourier transform to momentum space for kinetic term
    eta_tilde = np.fft.fftn(eta)
    # Define the Fourier space wave numbers
    k = np.fft.fftfreq(N, d=epsilon) * 2 * np.pi  # FFT frequencies, scaled
    k_mesh = np.meshgrid(*([k] * dim), indexing='ij')  # Create a meshgrid for each dimension

    # Calculate eigenvalues of K_hat in Fourier space using the known formula
    K_eigenvalues = sum((2 / (mu * epsilon**2)) * np.sin(k_dim / 2)**2 for k_dim in k_mesh)

    # Define the kinetic evolution operator in Fourier space
    def kinetic_evolution_operator(tau_hat):
        return np.exp(-1j * tau_hat * K_eigenvalues)

    K_exp = kinetic_evolution_operator(tau_hat)

    #apply kinetic part
    xi = np.fft.ifftn(K_exp * eta_tilde) #transform back to position space

    return V_half * xi