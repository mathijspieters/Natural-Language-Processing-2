import numpy as np

def interpolation(n_steps, latent_size):
    row = []
    a = np.random.normal(size=latent_size)
    b = np.random.normal(size=latent_size)

    for s in np.linspace(0, 1, n_steps-2):
        row.append( a*np.sqrt(s) + b*np.sqrt(1-s))

    row = np.array([b] + row + [a]).reshape((n_steps), latent_size)

    return row