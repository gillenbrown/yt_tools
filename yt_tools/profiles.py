import numpy as np

def plummer_3d(r, M, a):
    coeff = (3 * M) / (4 * np.pi * a**3)
    radii_term = (1 + (r/a)**2)**(-5.0 / 2.0)
    return coeff * radii_term

def plummer_2d(r, M, a):
    return M / (np.pi * a**2.0 * (1.0 + (r/a)**2)**2)

def exp_disk(r, M, a):
    return (M / (2 * np.pi * a**2)) * np.exp(-r/a)

def hernquist_3d(r, M, a):
    return M * a / (2.0 * np.pi * r * (r + a)**3)

def arcsech(x):
    return np.log((1 + np.sqrt(1.0 - x**2)) / x)

def hernquist_x(s):
    if s == 1:
        return 1
    elif s == 0:
        return np.log(2.0 / s)
    elif s < 1:
        return arcsech(s) / (np.sqrt(1 - s**2))
    else:
        return np.arccos(1.0 / s) / np.sqrt(s**2 - 1)

hernquist_x_vectorized = np.vectorize(hernquist_x)

def hernquist_2d(r, M, a):
    s = r / a
    if s == 1:
        return 2*M / (15 * np.pi * a**2)
    x = hernquist_x_vectorized(s)
    return M * ((2 + s**2) * x - 3) / (2 * np.pi * a**2 * (1 - s**2)**2)

hernquist_2d = np.vectorize(hernquist_2d)
