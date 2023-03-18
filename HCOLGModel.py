from types import SimpleNamespace
import time
import numpy as np
from scipy import optimize

class OLGModelClass():

    def __init__(self,do_print=True):
        """ create the model. """

        if do_print: print('initializing the model:')
        # use namespace to contain information about the model
        self.par = SimpleNamespace()
        self.sim = SimpleNamespace()

        if do_print: print('calling .setup()')
        self.setup()

        if do_print: print('calling .allocate()')
        self.allocate()
    
    def setup(self):
        """ baseline parameters. """
        # unpacks parameters
        par = self.par

        # set parameters
        # a. household
        par.sigma = 2.0 # CRRA coefficient
        par.beta = 1/1.40 # discount factor

        # b. firms
        par.production_function = 'ces'
        par.alpha = 0.30 # capital weight
        par.theta = 0.05 # substitution parameter
        par.delta = 0.50 # depreciation rate

        # c. government
        par.tau_w = 0.10 # labor income tax
        par.tau_r = 0.20 # capital income tax

        # d. misc
        par.K_lag_ini = 1.0 # initial capital stock
        par.B_lag_ini = 0.0 # initial government debt
        par.simT = 50 # length of simulation

    def allocate(self):
        """ allocate arrays for simulation. """
        
        par = self.par
        sim = self.sim

        # a. list of variables
        household = ['C1','C2']
        firm = ['K','Y','K_lag']
        prices = ['w','rk','rb','r','rt']
        government = ['G','T','B','balanced_budget','B_lag']

        # b. allocate space for all variables
        allvarnames = household + firm + prices + government # <-- add list together
        for varname in allvarnames:
        #__dict__[key] dictionary used to store an object's attributes
            sim.__dict__[varname] = np.nan*np.ones(par.simT) # create vector with shape (50, )      

    def simulate(self,do_print=True):
        """ simulate model. """
    
        t0 = time.time()
        
        # unpack
        par = self.par
        # assign attribute
        sim = self.sim
        
        # a. initial values
        sim.K_lag[0] = par.K_lag_ini
        sim.B_lag[0] = par.B_lag_ini

        # b. iterate
        for t in range(par.simT):
            
            # i. simulate before s
            simulate_before_s(par,sim,t)

            if t == par.simT-1: continue  # <-- if last period stops with iteration

            # i. find bracket to search in 
            s_min,s_max = find_s_bracket(par,sim,t)

            # ii. find optimal s using bisection search
            obj = lambda s: calc_euler_error(s,par,sim,t=t) #  <-- create anonymous function based on input s
            result = optimize.root_scalar(obj,bracket=(s_min,s_max),method='bisect')
            s = result.root

            # iii. simulate after s
            simulate_after_s(par,sim,t,s)

        if do_print: print(f'simulation done in {time.time()-t0:.2f} secs')

def find_s_bracket(par,sim,t,maxiter=500,do_print=False):
    """ find bracket for s to search in. """

    # a. maximum bracket
    s_min = 0.0 + 1e-8 # save almost nothing
    s_max = 1.0 - 1e-8 # save almost everything

    # b. saving a lot is always possible 
    value = calc_euler_error(s_max,par,sim,t)
    sign_max = np.sign(value)
    if do_print: print(f'euler-error for s = {s_max:12.8f} = {value:12.8f}')

    # c. find bracket      
    lower = s_min
    upper = s_max

    it = 0
    while it < maxiter:
                
        # i. midpoint and value
        s = (lower+upper)/2 # midpoint
        value = calc_euler_error(s,par,sim,t)

        if do_print: print(f'euler-error for s = {s:12.8f} = {value:12.8f}')

        # ii. check conditions
        valid = not np.isnan(value)
        correct_sign = np.sign(value)*sign_max < 0
        
        # iii. next step
        if valid and correct_sign: # found!
            s_min = s
            s_max = upper
            if do_print: 
                print(f'bracket to search in with opposite signed errors:')
                print(f'[{s_min:12.8f}-{s_max:12.8f}]')
            return s_min,s_max
        elif not valid: # too low s -> increase lower bound
            lower = s
        else: # too high s -> increase upper bound
            upper = s

        # iv. increment
        it += 1

    raise Exception('cannot find bracket for s')

def calc_euler_error(s,par,sim,t):
    """ target function for finding s with bisection """

    # a. simulate forward
    simulate_after_s(par,sim,t,s) # <-- simulate what happens after choosing s
    simulate_before_s(par,sim,t+1) # <-- next period simulate what happens before choosing s

    # c. Euler equation
    LHS = sim.C1[t]**(-par.sigma)
    RHS = (1+sim.rt[t+1])*par.beta * sim.C2[t+1]**(-par.sigma)

    return LHS-RHS

def simulate_before_s(par,sim,t):
    """ simulate forward. """

    # if t greater than zero lagged value is just value in previous period
    if t > 0: 
        sim.K_lag[t] = sim.K[t-1]
        sim.B_lag[t] = sim.B[t-1]

    # basically below is an implimentation of the conditions under section "simulation algorithm" in model text

    # a. production and factor prices
    # defining our production function based on general (CES) or special type (cobb-douglas)
    if par.production_function == 'ces':

        # i. production
        sim.Y[t] = ( par.alpha*sim.K_lag[t]**(-par.theta) + (1-par.alpha)*(1.0)**(-par.theta) )**(-1.0/par.theta)

        # ii. factor prices
        sim.rk[t] = par.alpha*sim.K_lag[t]**(-par.theta-1) * sim.Y[t]**(1.0+par.theta)
        sim.w[t] = (1-par.alpha)*(1.0)**(-par.theta-1) * sim.Y[t]**(1.0+par.theta)

    elif par.production_function == 'cobb-douglas':

        # i. production
        sim.Y[t] = sim.K_lag[t]**par.alpha * (1.0)**(1-par.alpha)

        # ii. factor prices
        sim.rk[t] = par.alpha * sim.K_lag[t]**(par.alpha-1) * (1.0)**(1-par.alpha)
        sim.w[t] = (1-par.alpha) * sim.K_lag[t]**(par.alpha) * (1.0)**(-par.alpha)

    else:

        raise NotImplementedError('unknown type of production function')

    # b. no-arbitrage and after-tax return
    sim.r[t] = sim.rk[t]-par.delta # after-depreciation return
    sim.rb[t] = sim.r[t] # same return on bonds
    sim.rt[t] = (1-par.tau_r)*sim.r[t] # after-tax return

    # c. consumption
    sim.C2[t] = (1+sim.rt[t])*(sim.K_lag[t]+sim.B_lag[t])

    # d. government
    sim.T[t] = par.tau_r*sim.r[t]*(sim.K_lag[t]+sim.B_lag[t]) + par.tau_w*sim.w[t]

    if sim.balanced_budget[t]:
        sim.G[t] = sim.T[t] - sim.r[t]*sim.B_lag[t]

    sim.B[t] = (1+sim.r[t])*sim.B_lag[t] - sim.T[t] + sim.G[t]

def simulate_after_s(par,sim,t,s):
    """ simulate forward """

    # a. consumption of young
    sim.C1[t] = (1-par.tau_w)*sim.w[t]*(1.0-s)

    # b. end-of-period stocks
    I = sim.Y[t] - sim.C1[t] - sim.C2[t] - sim.G[t]
    sim.K[t] = (1-par.delta)*sim.K_lag[t] + I