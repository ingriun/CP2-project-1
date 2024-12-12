import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
from subproject1 import strang_splitting_integrator, ndim_Ones, potential
from subproject1 import N, mu, epsilon, tau_hat, dim

def initialWavepacket(dim, N):
    x_positions = np.arange(N**dim)
    psi = ndim_Ones(dim, N)

    psi =  np.exp(-(x_positions-220)**2/(2*5**2))


    plt.plot(x_positions, psi.flatten())
    plt.title("Initial Wave Packet")
    plt.xlabel("Position")
    plt.xlim(0,300)
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
    
    # The plot will show the magnitude of psi at each step
    line, = ax.plot(x_positions, (psi.flatten()), 'lightgrey', label="Initial Ψ")
    ax.plot(x_positions, potential(psi.shape))
    ax.set_ylim(0, 1.5 * np.max(np.abs(psi)))
    ax.set_xlim(0,500)
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
        
        # Flatten psi 
        psi_flat = (psi.flatten())
        
        # Update the plot 
        line.set_ydata(np.abs(psi_flat))  # Plot the magnitude of psi
        line.set_color('g')
        return line,

    # Create the animation
    ani = animation.FuncAnimation(fig, update_frame, frames=times, interval=10, blit=True)
    
    plt.show()


animate_wave_function(dim = 1, N = N, num_frames=1000, integrator=strang_splitting_integrator)


