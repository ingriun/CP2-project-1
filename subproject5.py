import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
#from subproject2 import ndim_Random
from subproject1 import strang_splitting_integrator, ndim_Ones, potential
from subproject1 import N, mu, epsilon, tau_hat, dim

def initialWavepacket(dim, N):
    x_positions = np.arange(N**dim)
    psi = ndim_Ones(dim, N)
    #wavelenght = random.uniform(0.1 , 0.3)
    wavelength = 20
    L = N
    print('wl: ', wavelength)
    #k = 2*np.pi / wavelength
    #A = 1 / np.sqrt(2 * L)
    #B = A
    c = 3 * 10**8
    psi = 1 / (np.sqrt(2 * L)) * np.exp(-(x_positions-20)**2/(2*5**2))
    #for index in np.ndindex(psi.shape):
        #psi[index] = A * np.exp(1j * k * (x_positions[index])) + B * np.exp(1j * k * (x_positions[index]))
        #psi[index]=np.exp(-(x_positions[index]-c*tau_hat)**2)*(np.cos(2*np.pi*(x_positions[index]-c*tau_hat)/wavelength) + 1j*np.sin(2*np.pi*(x_positions[index]-c*tau_hat)/wavelength))
        #psi[index]=np.exp(-(x_positions[index]-50)**2/(2*20**2))

    plt.plot(x_positions, np.abs(psi.flatten()))
    plt.title("Initial Wave Packet")
    plt.xlabel("Position")
    plt.ylabel("|ψ|")
    plt.show()
    return x_positions, psi



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
    line, = ax.plot(x_positions, (psi.flatten()), 'lightgrey', label="Initial Ψ")
    ax.plot(x_positions, potential(psi))
    ax.set_ylim(0, 1.5 * np.max(np.abs(psi)))
    ax.set_xlim(0,100)
    ax.set_title("Time Evolution of the Wave Function")
    ax.set_xlabel("Position")
    ax.set_ylabel("Ψ")
    ax.legend()
    
    
    def update_frame(frame):
        """
        Update the wave function at each frame.
        """
        # Apply the integrator to evolve the wave function
        nonlocal psi
        psi = integrator(psi, tau_hat)
        print(f"Frame {frame}: psi = {psi}")
        # Flatten psi 
        psi_flat = (psi.flatten())
        
        # Update the plot 
        line.set_ydata(np.abs(psi_flat))  # Plot the magnitude of psi
        line.set_color('g')
        return line,

    # Create the animation
    ani = animation.FuncAnimation(fig, update_frame, frames=times, interval=500, blit=True)
    
    plt.show()


# Example usage:
animate_wave_function(dim = 1, N = N, num_frames=1000, integrator=strang_splitting_integrator)


