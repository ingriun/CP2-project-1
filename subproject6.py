
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from subproject2 import ndim_Random
from subproject1 import strang_splitting_integrator, tau_hat, ndim_Ones
from subproject1 import N, mu, epsilon, tau_hat, dim

# Initialize the wavepacket
def initialWavepacket(dim, N):
    x_positions = np.arange(N**dim)
    psi = ndim_Ones(dim, N)
    #wavelenght = random.uniform(0.1 , 0.3)
    wavelength = 0.21
    print('wl: ', wavelength)
    k = 2*np.pi / wavelength
    A = 1
    B = 2
    c = 3 * 10**8
    for index in np.ndindex(psi.shape):
        psi[index] = A * np.exp(1j * k * (x_positions[index])) + B * np.exp(1j * k * (x_positions[index]))

    return x_positions, psi

# Define the potential barrier
def potential_barrier(dim, N):
    barrier = np.zeros((N**dim,))
    x_positions = np.linspace(0, 1, N**dim)

    # Place the barrier in the middle of the domain
    center = 0.5
    width = 0.05
    height = 2  # Barrier height
    barrier = height * np.exp(-((x_positions - center) / width) ** 2)  # Gaussian barrier

    return barrier

# Animate the wavefunction with tunneling
def animate_wave_function_tunneling(dim, N, num_frames=100, integrator=strang_splitting_integrator):
    x_positions, psi = initialWavepacket(dim, N)
    potential = potential_barrier(dim, N)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Time steps to animate
    times = np.arange(num_frames)

    # Plot the potential barrier
    ax.plot(x_positions, potential, 'r--', label="Potential Barrier")

    # Plot the wavefunction
    line, = ax.plot(x_positions, np.abs(psi.flatten()), 'lightgrey', label="Magnitude |Ψ|")
    ax.set_ylim(0, 12)  # Adjust for the barrier's height
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
        psi = integrator(psi, tau_hat, potential)  # Modified integrator to include potential

        # Update the plot
        psi_flat = (psi.flatten())
        line.set_ydata(psi_flat)
        return line,

    # Create the animation
    ani = animation.FuncAnimation(fig, update_frame, frames=times, interval=300, blit=True)
    plt.show()

# Example usage:
animate_wave_function_tunneling(dim=1, N=100, num_frames=1000, integrator=strang_splitting_integrator)
