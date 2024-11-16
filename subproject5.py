import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
from subproject2 import ndim_Random
from subproject1 import strang_splitting_integrator, tau_hat, ndim_Ones
from subproject1 import N, mu, epsilon, tau_hat, dim

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


"""x_positions, psi = initialWavepacket(dim = 1, N = 100)
fig, ax = plt.subplots(figsize=(8, 6))
line, = ax.plot(x_positions, (psi.flatten()), label="Magnitude |Ψ|")
#ax.set_ylim(0, 1)
ax.set_title("Time Evolution of the Wave Function")
ax.set_xlabel("Position")
ax.set_ylabel("Magnitude of Wave Function")
plt.show()"""

def animate_wave_function(dim, N, num_frames=100, integrator=strang_splitting_integrator):
    # Initialize psi
    x_positions, psi = initialWavepacket(dim, N)
    #psi = ndim_Random(dim, N)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Time steps to animate
    times = np.arange(num_frames)
    
    # Define the x-axis positions for a flattened grid of N x N 
    #x_positions = np.arange(N**dim)  # Flattened 1D array of positions corresponding to psi
    
    # The plot will show the magnitude of psi at each step
    line, = ax.plot(x_positions, (psi.flatten()), 'lightgrey', label="Magnitude |Ψ|")
    ax.set_ylim(-5, 5)
    ax.set_title("Time Evolution of the Wave Function")
    ax.set_xlabel("Position")
    ax.set_ylabel("Magnitude of Wave Function")
    
    
    def update_frame(frame):
        """
        Update the wave function at each frame.
        """
        # Apply the integrator to evolve the wave function
        nonlocal psi
        psi = integrator(psi, tau_hat)
        
        # Flatten psi 
        psi_flat = (psi.flatten())
        
        # Update the plot 
        line.set_ydata(psi_flat)  # Plot the magnitude of psi
        line.set_color('g')
        return line,

    # Create the animation
    ani = animation.FuncAnimation(fig, update_frame, frames=times, interval=300, blit=True)
    
    plt.show()

# Example usage:
animate_wave_function(dim, N, num_frames=1000, integrator=strang_splitting_integrator)