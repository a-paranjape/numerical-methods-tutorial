import numpy as np

from ode import ODEInt

class Test_ODEInt(ODEInt):
    """ Implement convergence tests for ODEInt. """
    #########################################
    def __init__(self,setup={},ode_inst=None):
        """ Initialize class.
            -- setup: dictionary for initializing ODEInt
            -- ode_inst: instance of ODE class (such as those in examples.py)
        """
        if ode_inst is None:
            raise Exception("ode_inst must be specified in Test_ODEInt.")
        self.ode_inst = ode_inst
        ODEInt.__init__(self,setup=setup)

        self.etype_dict = {'solution':'u_exact',
                           'velocity':'v_exact',
                           'energy':'energy'}
    #########################################


    #########################################
    def global_error(self,T,dt,u0,v0,theta_f=None,theta_v=None,error_type='energy'):
        """ Calculate global error on solution. """
        if error_type not in self.etype_dict.keys():
            raise Exception("error_type must be one of ["+','.join([e for e in list(self.etype_dict.keys())])+"].")
        
        if not hasattr(self.ode_inst,self.etype_dict[error_type]):
            raise Exception(error_type+" error cannot be calculated since "+self.etype_dict[error_type]+" doesn't exist in ODE instance")
        
        u,v,t = self.solver(T,dt,u0,v0,theta_f=theta_f,theta_v=theta_v)
        if error_type in ['solution','velocity']:
            exact = getattr(self.ode_inst,self.etype_dict[error_type])(t,u0,v0,theta_f=theta_f,theta_v=theta_v)
            error = exact - u if error_type=='solution' else exact - v # running residual
            error = np.sqrt(np.sum(error**2)*dt) # global error
        elif error_type == 'energy':
            energy = self.ode_inst.energy(t,u,v,theta_f=theta_f,theta_v=theta_v)
            error = np.abs(energy/(energy[0]+1e-15) - 1).max() # l-inf norm of relative energy residual
            
        return error
    #########################################

    #########################################
    def convergence(self,T,u0,v0,theta_f=None,theta_v=None,dt_min=1e-4,dt_max=1.0,n_evals=10,error_type='energy'):
        """ Calculate convergence rate for global solution error. """
        if error_type not in self.etype_dict.keys():
            raise Exception("error_type must be one of ["+','.join([e for e in list(self.etype_dict.keys())])+"].")
        
        if not hasattr(self.ode_inst,self.etype_dict[error_type]):
            raise Exception(error_type+" error cannot be calculated since "+self.etype_dict[error_type]+" doesn't exist in ODE instance")

        dts = np.logspace(np.log10(dt_min),np.log10(dt_max),n_evals)
        dts = dts[::-1]
        dlnDt = np.log(dts[1]/dts[0])
        rate = np.zeros_like(dts)
        error = np.zeros_like(dts)
        error[0] = self.global_error(T,dts[0],u0,v0,theta_f=theta_f,theta_v=theta_v,error_type=error_type)
        for t in range(1,dts.size):
            error[t] = self.global_error(T,dts[t],u0,v0,theta_f=theta_f,theta_v=theta_v,error_type=error_type)
            rate[t] = np.log(error[t]/error[t-1])/dlnDt
        
        return error[1:],rate[1:],dts[1:]
    #########################################


    
    # #########################################
    # def solution_error(self,T,dt,u0,v0,theta_f=None,theta_v=None):
    #     """ Calculate global error on solution. """
    #     if not hasattr(self.ode_inst,'u_exact'):
    #         raise Exception("solution error cannot be calculated since exact solution of ODE doesn't exist")
        
    #     u,v,t = self.solver(T,dt,u0,v0,theta_f=theta_f,theta_v=theta_v)
    #     u_exact = self.ode_inst.u_exact(t,u0,v0,theta_f=theta_f,theta_v=theta_v)
    #     error = u_exact - u # running residual
    #     error = np.sqrt(np.sum(error**2)*dt) # global error
    #     return error
    # #########################################

    # #########################################
    # def convergence_solution(self,T,u0,v0,theta_f=None,theta_v=None,dt_min=1e-4,dt_max=1.0,n_evals=10):
    #     """ Calculate convergence rate for global solution error. """
    #     if not hasattr(self.ode_inst,'u_exact'):
    #         raise Exception("solution error convergence cannot be calculated since exact solution of ODE doesn't exist")

    #     dts = np.logspace(np.log10(dt_min),np.log10(dt_max),n_evals)
    #     dts = dts[::-1]
    #     dlnDt = np.log(dts[1]/dts[0])
    #     rate = np.zeros_like(dts)
    #     error = np.zeros_like(dts)
    #     error[0] = self.solution_error(T,dts[0],u0,v0,theta_f=theta_f,theta_v=theta_v)
    #     for t in range(1,dts.size):
    #         error[t] = self.solution_error(T,dts[t],u0,v0,theta_f=theta_f,theta_v=theta_v)
    #         rate[t] = np.log(error[t]/error[t-1])/dlnDt
        
    #     return error[1:],rate[1:],dts[1:]
    # #########################################

    # #########################################
    # def velocity_error(self,T,dt,u0,v0,theta_f=None,theta_v=None):
    #     """ Calculate global error on velocity. """
    #     if not hasattr(self.ode_inst,'v_exact'):
    #         raise Exception("velocity error cannot be calculated since exact velocity of ODE doesn't exist")
        
    #     u,v,t = self.solver(T,dt,u0,v0,theta_f=theta_f,theta_v=theta_v)
    #     v_exact = self.ode_inst.v_exact(t,u0,v0,theta_f=theta_f,theta_v=theta_v)
    #     error = v_exact - v # running residual
    #     error = np.sqrt(np.sum(error**2)*dt) # global error
    #     return error
    # #########################################

    # #########################################
    # def convergence_velocity(self,T,u0,v0,theta_f=None,theta_v=None,dt_min=1e-4,dt_max=1.0,n_evals=10):
    #     """ Calculate convergence rate for global velocity error. """
    #     if not hasattr(self.ode_inst,'v_exact'):
    #         raise Exception("velocity error convergence cannot be calculated since exact velocity of ODE doesn't exist")

    #     dts = np.logspace(np.log10(dt_min),np.log10(dt_max),n_evals)
    #     dts = dts[::-1]
    #     dlnDt = np.log(dts[1]/dts[0])
    #     rate = np.zeros_like(dts)
    #     error = np.zeros_like(dts)
    #     error[0] = self.velocity_error(T,dts[0],u0,v0,theta_f=theta_f,theta_v=theta_v)
    #     for t in range(1,dts.size):
    #         error[t] = self.solution_error(T,dts[t],u0,v0,theta_f=theta_f,theta_v=theta_v)
    #         rate[t] = np.log(error[t]/error[t-1])/dlnDt
        
    #     return error[1:],rate[1:],dts[1:]
    # #########################################

    # #########################################
    # def energy_error(self,T,dt,u0,v0,theta_f=None,theta_v=None):
    #     """ Calculate maximum error on energy. """
    #     if not hasattr(self.ode_inst,'energy'):
    #         raise Exception("energy error cannot be calculated since energy of ODE doesn't exist")
        
    #     u,v,t = self.solver(T,dt,u0,v0,theta_f=theta_f,theta_v=theta_v)
    #     energy = self.ode_inst.energy(t,u,v,theta_f=theta_f,theta_v=theta_v)
    #     error = np.abs(energy/(energy[0]+1e-15) - 1).max() # l-inf norm of relative energy residual
    #     return error
    # #########################################

    # #########################################
    # def convergence_energy(self,T,u0,v0,theta_f=None,theta_v=None,dt_min=1e-4,dt_max=1.0,n_evals=10):
    #     """ Calculate convergence rate for energy error. """
    #     if not hasattr(self.ode_inst,'energy'):
    #         raise Exception("energy error convergence cannot be calculated since energy of ODE doesn't exist")

    #     dts = np.logspace(np.log10(dt_min),np.log10(dt_max),n_evals)
    #     dts = dts[::-1]
    #     dlnDt = np.log(dts[1]/dts[0])
    #     rate = np.zeros_like(dts)
    #     error = np.zeros_like(dts)
    #     error[0] = self.energy_error(T,dts[0],u0,v0,theta_f=theta_f,theta_v=theta_v)
    #     for t in range(1,dts.size):
    #         error[t] = self.solution_error(T,dts[t],u0,v0,theta_f=theta_f,theta_v=theta_v)
    #         rate[t] = np.log(error[t]/error[t-1])/dlnDt
        
    #     return error[1:],rate[1:],dts[1:]
    # #########################################
