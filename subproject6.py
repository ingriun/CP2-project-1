import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from subproject1 import hamiltonian, laplacian, kineticEnergy, strang_splitting_integrator, second_order_integrator, ndim_Ones, ndim_Random
from subproject1 import N, mu, epsilon, tau_hat, dim

# Gaussian potential function
def gaussian_potential(psi, width=8, height=10, center=N//2, shift=2.5):
    """
    Create a Gaussian potential barrier centered in the middle of the array.
    Parameters:
    - psi: Wavefunction to determine the grid size.
    - width: Controls the width of the Gaussian barrier.
    - height: Controls the height of the potential.
    - center: The position of the center of the barrier.
    
    Returns:
    - v_hat: The Gaussian potential array.
    """
    N = psi.shape[0]  # Get the grid size from the shape of the wavefunction
    coordinates = np.linspace(-N//2, N//2, N)
    
    v_hat = height * np.exp(-0.5 * (coordinates**2 / width**2))  # Gaussian shape
    v_hat1 = v_hat - shift
    
    return v_hat1

# Hamiltonian with Gaussian potential
def hamiltonian_with_gaussian_potential(psi):
    psi_2nd = laplacian(psi)
    v_hat = gaussian_potential(psi)
    k_hat = kineticEnergy(psi)
    h_hat = k_hat*psi_2nd + v_hat * psi
    return h_hat


# Initialize the wavepacket
def initialWavepacket(dim, N):
    x_positions = np.arange(N**dim)
    psi = ndim_Ones(dim, N)
    wavelength = 0.21
    print('wl: ', wavelength)
    k = 2*np.pi / wavelength
    A = 1
    B = 0.5
    c = 3 * 10**8
    """center = N**dim // 5  # Center of the Gaussian envelope
    width = N**dim / 20   # Width of the Gaussian envelope (adjust as needed)"""
    for index in np.ndindex(psi.shape):
        psi[index] = A * np.exp(1j * k * (x_positions[index])) + B * np.exp(1j * k * (x_positions[index]))
    """for index in np.ndindex(psi.shape):
        x = x_positions[index]  # Current position
        envelope = np.exp(-((x - center)**2) / (2 * width**2))  # Gaussian envelope
        psi[index] = A * envelope * np.exp(1j * k * x)  # Wave with Gaussian envelope"""

    return x_positions, psi


def strang_splitting_integrator(psi, tau_hat, gaussian_potential):
    # Split Hamiltonian into kinetic and potential parts
    v_half = np.exp(-1j * (tau_hat / 2) * gaussian_potential(psi))  # e^(-i*tau_hat/2 * V)
    
    # Apply potential
    eta = v_half * psi

    # Fourier transform to momentum space
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

    # Apply kinetic part
    xi = np.fft.ifftn(k_exp * eta_tilde)  # Transform back to position space

    return v_half * xi

# Animate the wavefunction with tunneling
def animate_wave_function_tunneling(dim, N, num_frames=100, integrator=strang_splitting_integrator):
    x_positions, psi = initialWavepacket(dim, N)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Time steps to animate
    times = np.arange(num_frames)

    # Create the potential based on the wavefunction
    potential = gaussian_potential(psi)  # Now passing psi to the potential function

    # Plot the potential barrier
    ax.plot(x_positions, potential, 'r--', label="Potential Barrier")

    # Plot the wavefunction
    line, = ax.plot(x_positions, np.abs(psi.flatten()), 'lightgrey', label="Magnitude |Ψ|")
    ax.set_ylim(-4, 6)  # Adjust for the barrier's height
    ax.set_xlim(0, 100)
    ax.set_title("Wavepacket Tunneling through a Barrier")
    ax.set_xlabel("Position")
    ax.set_ylabel("Magnitude of Wave Function")
    ax.legend()
    
    def update_frame(frame):
        """
        Update the wave function at each frame.
        """
        nonlocal psi
        # Add potential effect into the integrator
        psi = integrator(psi, tau_hat, gaussian_potential)  # Pass gaussian_potential directly

        # Update the plot
        psi_flat = (psi.flatten())
        line.set_ydata(psi_flat)
        return line,

    # Create the animation
    ani = animation.FuncAnimation(fig, update_frame, frames=times, interval=300, blit=True)
    plt.show()

# Example usage:
animate_wave_function_tunneling(dim=1, N=201, num_frames=1000, integrator=strang_splitting_integrator)
