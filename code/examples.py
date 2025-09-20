import numpy as np

# Simple harmonic oscillator
class SimpleHarmonicOscillator(object):
    def f_ut(self,u,t,theta_f):
        return -theta_f[0]**2*u

    def u_exact(self,t,u0,v0,theta_f,theta_v=None):
        return u0*np.cos(theta_f[0]*t) + (v0/theta_f[0])*np.sin(theta_f[0]*t)

    def v_exact(self,t,u0,v0,theta_f,theta_v=None):
        return -u0*theta_f[0]*np.sin(theta_f[0]*t) + v0*np.cos(theta_f[0]*t)

    def energy(self,t,u,v,theta_f,theta_v=None):
        E = 0.5*(v**2 + (theta_f[0]*u)**2)
        return E
