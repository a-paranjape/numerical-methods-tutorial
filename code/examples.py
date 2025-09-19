import numpy as np

# Simple harmonic oscillator
def f_ut_SHO(u,t,theta_f):
    return -theta_f[0]**2*u
