import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
import random

from IPython.display import HTML, clear_output

# helpful to raise exception upon complex number calculation error
# instead of just a useless warning
import warnings
warnings.filterwarnings(action="error", category=np.ComplexWarning)

def storPsi(E, psi, c, t):
    total = 0
    for n in range(E.size):
        total += c[n] * psi[n] * np.exp(-1j * E[n] * t / hbar)
    return total

def animate_wave(x, v, wave0, fps, koeff, psi, E, t1=0, t2=None, realtime=True, real=False, imag=False):
    assert not (not realtime and t2 is None), "non-realtime animation must be finite in time"

    dt = 1 / fps
    nframes = None if t2 is None else int((t2 - t1) / dt) # None animates forever
    
    # print information about this animation
    nframesstr = "infinite" if t2 is None else f"{nframes}"
    durationstr = "infinite" if t2 is None else f"{t2-t1:.2f}"
    print("Animation information:")
    print(f"  Frames   : {nframesstr}")
    print(f"  Framerate: {fps} FPS")
    print(f"  Duration : {durationstr}")
    print(f"  Time step: {dt}")
    
    fig, ax = plt.subplots()
    
    # prevent duplicate animation figure in non-realtime mode
    #if not realtime:
        #clear_output()
        
    ax.set_xlabel("$x$")
    ax.set_ylabel("$|\Psi|$, $\Re{(\Psi)}$, $\Im{(\Psi)}$")
    
    # create objects for the graphs that will be updated every frame
    # the commas matter!
    ymax = np.max(np.abs(wave0))
    graph, = ax.plot([x[0], x[-1]], [0, +2*ymax]) # plot 2x wave0 to make room
    if real:
        graph2, = ax.plot([x[0], x[-1]], [0, -2*ymax]) # make room for negative values
    if imag:
        graph3, = ax.plot([x[0], x[-1]], [0, -2*ymax]) # make room for negative values
    
    # plot potential extended with indications of infinite walls at ends
    ax2 = ax.twinx()
    v_max = np.min(v) + 1.1 * (np.max(v) - np.min(v)) + 1 # + 1 if v = const
    x_ext = np.concatenate(([x[0]], x, [x[-1]]))
    v_ext = np.concatenate(([v_max], v, [v_max]))
    ax2.set_ylabel("$V(x)$")
    ax2.plot(x_ext, v_ext, linewidth=3, color="black", label="V")
    ax2.legend(loc="upper right")
    
    # call this function for every frame in the animation
    def animate(i):
        time = t1 + i*dt
        wave = storPsi(E, psi, koeff, time)

        
        # update graph objects
        # set_data() etc. modifies existing an existing object in a figure
        # it is much more efficient than creating a new figure for every animation frame
        graph.set_data(x, np.abs(wave))
        graph.set_label(f"$|\Psi(x, t = {time:.2f})|$")
        if real:
            graph2.set_data(x, np.real(wave))
            graph2.set_label(f"$\Re(\Psi(x, t = {time:.2f}))$")
        if imag:
            graph3.set_data(x, np.imag(wave))
            graph3.set_label(f"$\Im(\Psi(x, t = {time:.2f}))$")     
        ax.legend(loc="upper left")

    # create matplotlib animation object
    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=nframes, interval=dt*1000, repeat=False)   
    if realtime:
        return ani
    else:
        return HTML(ani.to_jshtml())