import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
from subproject2 import ndim_Random
from subproject1 import strang_splitting_integrator

def animate_wave_function(dim, N, num_frames=100, integrator=strang_splitting_integrator):
    # Initialize psi
    psi = ndim_Random(dim, N)
    

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Time steps to animate
    times = np.arange(num_frames)
    
    # Define the x-axis positions for a flattened grid of N x N 
    x_positions = np.arange(N * N)  # Flattened 1D array of positions corresponding to psi
    
    # The plot will show the magnitude of psi at each step
    line, = ax.plot(x_positions, np.abs(psi.flatten()), label="Wave Function |psi|")
    ax.set_ylim(0, 1)
    ax.set_title("Time Evolution of the Wave Function")
    ax.set_xlabel("Position")
    ax.set_ylabel("Magnitude of Wave Function")
    
    def update_frame(frame):
        """
        Update the wave function at each frame.
        """
        # Apply the integrator to evolve the wave function
        global psi
        psi = integrator(psi)
        
        # Flatten psi 
        psi_flat = np.abs(psi.flatten())
        
        # Update the plot 
        line.set_ydata(psi_flat)  # Plot the magnitude of psi
        return line,

    # Create the animation
    ani = animation.FuncAnimation(fig, update_frame, frames=times, interval=50, blit=True)
    
    plt.show()

# Example usage:
animate_wave_function(dim=2, N=4, num_frames=100, integrator=strang_splitting_integrator)