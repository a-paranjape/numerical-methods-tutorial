import numpy as np
from utilities import Utilities # various utilities for simple i/o tasks
import inspect # useful for checking function call signatures

#############################################
class ODEInt(Utilities):
    """ Finite-differencing schema for 2nd-order ODE integration for generic problems
        of the type u'' = f(u,t,theta_f) + g(v,theta_v) where v = u'. 
    """
    #########################################
    def __init__(self,setup):
        """ Expect setup to be dictionary with keys being a subset of:
            -- 'scheme': string; integration scheme. Expect one of 
                         ['verlet','f-euler','euler-cromer','stoermer-verlet']
            -- 'f_ut': callable or None; function object with call signature f_ut(u,t,theta_f) where u,t are scalars 
                       and theta_f is array-like, containing parameters controlling f_ut (default None)
            -- 'g_v' : callable or None; function object with call signature g_v(g,theta_v) where v is a scalar 
                       and theta_v is array-like, containing parameters controlling g_v (default None)
            -- 'verbose': boolean; control verbosity of output (default True) 
            -- 'logfile': string or None; file to store verbose output, or None for output to stdio (default None)
        """
        # integration scheme: 
        self.allowed_schema = {'verlet':self.verlet,
                               'f_euler':self.f_euler,
                               'euler_cromer':self.euler_cromer,
                               'stoermer_verlet':self.stoermer_verlet}
        
        self.scheme = setup.get('scheme','verlet')
        self.solver = self.allowed_schema[self.scheme]

        self.f_ut = setup.get('f_ut',None) # function with call signature f_ut(u,t,theta_f)
        self.g_v = setup.get('g_v',None)   # function with call signature g_v(v,theta_v)

        self.f_is_None = (self.f_ut is None) # useful for handling a
        self.g_is_None = (self.g_v is None)  # few special use cases

        self.verbose = setup.get('verbose',True)
        self.logfile = setup.get('logfile',None)

        if self.verbose:
            self.print_this("---------------",self.logfile)
            self.print_this("ODE integration",self.logfile)
            self.print_this("---------------",self.logfile)
            self.print_this("... solving u''(t) = f(u,t) + g(v) with v = u'",self.logfile)
        
        self.check_init()
        if self.verbose:
            self.print_this("... setup complete",self.logfile)
            self.print_this("---------------",self.logfile)
    #########################################

    
    #########################################
    def check_init(self):
        """ Check validity of inputs. """
        if self.scheme not in self.allowed_schema.keys():
            raise Exception("scheme '"+self.scheme+"' not recognised in ODEInt. Expecting one of ["+','.join([s for s in list(self.allowed_schema.keys())])+']')

        if self.scheme is None:
            raise NotImplementedError("scheme '"+self.scheme+"' is not implemented yet!")

        if self.verbose:
            self.print_this("... using scheme: "+self.scheme,self.logfile)
        
        if self.g_v is None:
            # check that f_ut is defined if g_v is None
            if self.f_ut is None:
                raise Exception("f_ut and g_v cannot both be None in ODEInt.")

            if self.verbose:
                self.print_this("... g_v is None",self.logfile)
                
            # set g_v internally to be dummy function that returns 0.0 for any argument
            self.g_v = lambda v,theta_v: 0.0
        else:
            # if g_v is defined...
            # ... f_ut need not be defined
            if self.f_ut is None:
                # ... in which case, set f_ut internally to be dummy function that returns 0.0 for any argument
                self.f_ut = lambda u,t,theta_f: 0.0

            # ensure g_v is callable...
            if not hasattr(self.g_v,'__call__'):
                raise Exception("g_v must be callable")
            # ... and has the correct call signature
            if list(inspect.signature(self.g_v).parameters.keys()) != ['v','theta_v']:
                raise Exception("g_v must have call signature g_v(v,theta_v)")

        # having checked g_v, turn to f_ut if defined
        if self.f_ut is not None:
            # ensure f_ut is callable
            if not hasattr(self.f_ut,'__call__'):
                raise Exception("f_ut must be callable")
            # ... and has the correct call signature
            if list(inspect.signature(self.f_ut).parameters.keys()) != ['u','t','theta_f']:
                raise Exception("f_ut must have call signature f_ut(u,t,theta_f)")

        return
    #########################################

    
    #########################################
    def check_theta(self,theta_f,theta_v):
        """ Ensure function parameters theta_f and theta_v are supplied wherever needed. """
        if (not self.f_is_None) & (theta_f is None):
                raise Exception("theta_f must be specified as array-like for f_ut, even if not used (e.g. theta_f=[None])")
        if (not self.g_is_None) & (theta_v is None):
                raise Exception("theta_v must be specified as array-like for g_v, even if not used (e.g. theta_v=[None])")
        return
    #########################################


    #########################################
    def calc_uprime(self,u,dt,v0):
        """ Calculate v = u'(t) from u(t), using centered difference at all but boundary values. """
        v = np.zeros_like(u)
        v[0] = v0
        v[1:-1] = (u[2:] - u[:-2])/(2*dt)
        v[-1] = (u[-1] - u[-2])/dt
        return v
    #########################################


    #########################################
    def initialize_arrays(self,T,dt):
        """ Helper function to initialize u,v,t arrays. """
        
        Nt = int(round(T/dt)) # number of steps
        u = np.zeros(Nt+1,dtype=float) # solution array
        v = np.zeros(Nt+1,dtype=float) # velocity array
        t = np.linspace(0.0,Nt*dt,Nt+1) # time array
        
        return Nt,u,v,t
    #########################################

    
    #########################################
    # Solver methods
    #########################################
    def verlet(self,T,dt,u0,v0,theta_f=None,theta_v=None):
        """ Verlet integration. """

        if not self.g_is_None:
            raise Exception("Generic Verlet doesn't work with non-trivial g_v!")

        # check that all needed parameters are provided
        self.check_theta(theta_f,theta_v)

        # initialize arrays
        Nt,u,v,t = self.initialize_arrays(T,dt)
        dt2 = dt**2

        # set initial conditions
        u[0] = u0
        u[1] = u0 + dt*v0 + 0.5*dt2*self.f_ut(u0,t[0],theta_f)

        # recurse
        for n in range(1,Nt):
            u[n+1] = 2*u[n] - u[n-1] + dt2*self.f_ut(u[n],t[n],theta_f)

        # calculate v = u'
        v = self.calc_uprime(u,dt,v0)
            
        return u,v,t        
    #########################################


    #########################################
    def f_euler(self,T,dt,u0,v0,theta_f=None,theta_v=None):
        """ Forward-Euler scheme. """
        
        # check that all needed parameters are provided
        self.check_theta(theta_f,theta_v)

        # initialize arrays
        Nt,u,v,t = self.initialize_arrays(T,dt)

        # set initial conditions
        u[0] = u0
        v[0] = v0

        # recurse
        for n in range(Nt):
            v[n+1] = v[n] + dt*(self.f_ut(u[n],t[n],theta_f) + self.g_v(v[n],theta_v))
            u[n+1] = u[n] + dt*v[n] # cf. self.euler_cromer

        return u,v,t        
    #########################################


    #########################################
    def euler_cromer(self,T,dt,u0,v0,theta_f=None,theta_v=None):
        """ Euler-Cromer scheme. """
        
        # check that all needed parameters are provided
        self.check_theta(theta_f,theta_v)

        # initialize arrays
        Nt,u,v,t = self.initialize_arrays(T,dt)

        # set initial conditions
        u[0] = u0
        v[0] = v0

        # recurse
        for n in range(Nt):
            v[n+1] = v[n] + dt*(self.f_ut(u[n],t[n],theta_f) + self.g_v(v[n],theta_v))
            u[n+1] = u[n] + dt*v[n+1] # cf. self.f_euler

        return u,v,t        
    #########################################


    #########################################
    def stoermer_verlet(self,T,dt,u0,v0,theta_f=None,theta_v=None):
        """ Stoermer-Verlet scheme. """

        if not self.g_is_None:
            raise Exception("Generic Stoermer-Verlet doesn't work with non-trivial g_v!")

        # check that all needed parameters are provided
        self.check_theta(theta_f,theta_v)

        # initialize arrays
        Nt,u,v,t = self.initialize_arrays(T,dt)

        # internal v-indexing on staggered grid: v[n] = v(t_{n-1/2})
        
        # set initial conditions
        u[0] = u0
        v[0] = v0 - 0.5*dt*self.f_ut(u0,0.0,theta_f)

        # recurse
        for n in range(Nt):
            v[n+1] = v[n] + dt*self.f_ut(u[n],t[n],theta_f)
            u[n+1] = u[n] + dt*v[n+1] 

        # v on integer grid
        # ... first Nt steps (including t=0)
        v = 0.5*(v[1:] + v[:-1])
        # ... last step
        v_last = (u[-1]-u[-2])/dt
        v = np.concatenate((v,[v_last]))
            
        return u,v,t        
    #########################################
    
    
#############################################
