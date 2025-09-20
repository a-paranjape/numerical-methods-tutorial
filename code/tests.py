import numpy as np

from ode import ODEInt

class Test_ODEInt(ODEInt):
    """ Implement convergence tests for ODEInt. """
    def __init__(self,setup={},ode_inst=None):
        """ Initialize class.
            -- setup: dictionary for initializing ODEInt
            -- ode_inst: instance of ODE class (such as those in examples.py)
        """
        if ode_inst is None:
            raise Exception("ode_inst must be specified in Test_ODEInt.")
        self.ode_inst = ode_inst
        ODEInt.__init__(self,setup=setup)

    def solution_error(self,T,dt,u0,v0,theta_f=None,theta_v=None):
        """ Calculate global error on solution. """
        if not hasattr(self.ode_inst,'u_exact'):
            raise Exception("solution error cannot be calculated since exact solution of ODE doesn't exist")
        
        u,v,t = self.solver(T,dt,u0,v0,theta_f=theta_f,theta_v=theta_v)
        u_exact = self.ode_inst.u_exact(t,u0,v0,theta_f=theta_f,theta_v=theta_v)
        error = u_exact - u # running residual
        error = np.sqrt(np.sum(error**2)*dt) # global error
        return error

    def convergence_solution(self,T,u0,v0,theta_f=None,theta_v=None,dt_min=1e-4,dt_max=1.0,n_evals=10):
        """ Calculate convergence rate for global solution error. """
        if not hasattr(self.ode_inst,'u_exact'):
            raise Exception("solution error convergence cannot be calculated since exact solution of ODE doesn't exist")

        dts = np.logspace(np.log10(dt_min),np.log10(dt_max),n_evals)
        dts = dts[::-1]
        dlnDt = np.log(dts[1]/dts[0])
        rate = np.zeros_like(dts)
        error = np.zeros_like(dts)
        error[0] = self.solution_error(T,dts[0],u0,v0,theta_f=theta_f,theta_v=theta_v)
        for t in range(1,dts.size):
            error[t] = self.solution_error(T,dts[t],u0,v0,theta_f=theta_f,theta_v=theta_v)
            rate[t] = np.log(error[t]/error[t-1])/dlnDt
        
        return error[1:],rate[1:],dts[1:]
