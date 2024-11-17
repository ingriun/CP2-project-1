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
    wavelength = 0.22
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
    ax.set_xlim(0,100)
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

############################ Continuum and Infinite-Volume Checks ##########################

def continuum_limit_test(initial_psi_func, dim, resolutions, tau_hat, integrator):
    """
    Test convergence to the continuum limit by refining the lattice spacing (epsilon).
    """
    print("Continuum Limit Test:")
    errors = []
    reference_solution = None

    for i, N in enumerate(resolutions):
        # Update global variables
        epsilon = 0.03 * 101 / N

        # Generate initial wavefunction
        _, psi = initial_psi_func(dim, N)

        # Time-evolve wavefunction
        for _ in range(10):  # A fixed small number of time steps
            psi = integrator(psi, tau_hat)

        # Store the reference solution (finest resolution)
        if i == len(resolutions) - 1:
            reference_solution = psi.copy()

        # Compute error with reference solution if not the finest resolution
        if i > 0:
            error = np.linalg.norm(psi - reference_solution[:N]) / np.linalg.norm(reference_solution[:N])
            errors.append(error)

    # Plot the error against lattice spacing
    plt.figure(figsize=(8, 6))
    lattice_spacings = [0.03 * 101 / N for N in resolutions[:-1]]
    plt.loglog(lattice_spacings, errors, marker='o', label="Relative Error")
    plt.xlabel("Lattice Spacing (ε)")
    plt.ylabel("Relative Error")
    plt.title("Convergence to the Continuum Limit")
    plt.grid()
    plt.legend()
    plt.show()


def infinite_volume_limit_test(initial_psi_func, dim, volumes, tau_hat, integrator):
    """
    Test convergence to the infinite-volume limit by increasing the domain size (L).
    """
    print("Infinite Volume Limit Test:")
    boundary_amplitudes = []

    for N in volumes:
        # Update global variables
        epsilon = 0.03 * 101 / N

        # Generate initial wavefunction
        _, psi = initial_psi_func(dim, N)

        # Time-evolve wavefunction
        for _ in range(10):  # A fixed small number of time steps
            psi = integrator(psi, tau_hat)

        # Measure the amplitude of the wavefunction at the boundary
        boundary_amplitude = np.abs(psi[0]) + np.abs(psi[-1])
        boundary_amplitudes.append(boundary_amplitude)

    # Plot boundary amplitudes against domain size
    plt.figure(figsize=(8, 6))
    plt.plot(volumes, boundary_amplitudes, marker='o', label="Boundary Amplitude")
    plt.xlabel("Domain Size (N)")
    plt.ylabel("Boundary Amplitude")
    plt.title("Convergence to the Infinite-Volume Limit")
    plt.grid()
    plt.legend()
    plt.show()


def timestep_accuracy_test(initial_psi_func, dim, N, time_steps, integrator, tau_hat):
    """
    Test accuracy of the integrator by reducing the time step and comparing results.
    """
    print("Time Step Accuracy Test:")
    errors = []
    reference_solution = None

    # Generate initial wavefunction
    _, psi = initial_psi_func(dim, N)

    for i, tau in enumerate(time_steps):
        # Apply time evolution for the given time step
        evolved_psi = psi.copy()
        for _ in range(int(1 / tau)):  # Simulate 1 unit of time
            evolved_psi = integrator(evolved_psi, tau)

        # Store the reference solution (smallest time step)
        if i == len(time_steps) - 1:
            reference_solution = evolved_psi.copy()

        # Compute error with reference solution if not the finest time step
        if i > 0:
            error = np.linalg.norm(evolved_psi - reference_solution) / np.linalg.norm(reference_solution)
            errors.append(error)

    # Plot the error against time step size
    plt.figure(figsize=(8, 6))
    plt.loglog(time_steps[:-1], errors, marker='o', label="Relative Error")
    plt.xlabel("Time Step (τ)")
    plt.ylabel("Relative Error")
    plt.title("Time Step Accuracy")
    plt.grid()
    plt.legend()
    plt.show()


#################### Call Continuum and Infinite-Volume Tests ####################

# Example resolutions for continuum limit test
resolutions = [200]

# Example domain sizes for infinite-volume limit test
volumes = [200]

# Example time steps for timestep accuracy test
time_steps = [0.1]

# Call the tests
continuum_limit_test(initialWavepacket, dim, resolutions, tau_hat, strang_splitting_integrator)
infinite_volume_limit_test(initialWavepacket, dim, volumes, tau_hat, strang_splitting_integrator)
timestep_accuracy_test(initialWavepacket, dim, N, time_steps, strang_splitting_integrator, tau_hat)