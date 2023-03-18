from types import SimpleNamespace
import numpy as np
from scipy import optimize

class RamseyModelClass():
    
    def __init__(self,do_print=True): # note default argument has form 'keywordname=value'        
        """Create the model. """ 

        if do_print: print('initializing the model:') # every time we call the method it prints the text due to default argument do_print is set to True
       
        # create attributes by assigning a value, here we use SimpleNamespace (think of it as an effecicient dictionary)
        self.par = SimpleNamespace()
        self.ss = SimpleNamespace()
        self.path = SimpleNamespace()
        
        if do_print: print('calling .setup()')
        self.setup() # notice when calling other methods from within the class we use 'self' as prefix
        
        if do_print: print('calling .allocate()')
        self.allocate()
        
    def setup(self):
        """ baseline parameters. """
        # unpacks parameters
        par = self.par 
        
        # a. household 
        par.sigma = 2.0 # CRRA coefficient
        par.beta = np.nan # discount factor, which we assign to a placeholder 
        
        # b. firms
        par.A = np.nan # think of np.nan as a placeholder 
        par.production_function = 'cobb-douglas' # can be changed to ces!
        par.alpha = 0.30 # capital weight
        par.theta = 0.05 # substitution parameter
        par.delta = 0.05 # depreciation rate
        
        # c. initial value of k
        par.K_lag_ini = 1.0
        
        # d. misc 
        par.solver = 'broyden' # solver for the equation system, 'broyden' order 'scipy'
        
        par.Tpath = 500 # length of transition path, "truncation horizon"
        
    def allocate(self):
        """ allocate arrays (grid-like structure that holds data) for transition path """
        # picks parameters
        par = self.par
        # Create attribute path by assigning a value
        path = self.path
        
        # allocate space for all variables
        allvarnames = ['A','K','C','rk','w','r','Y','K_lag']
        for varname in allvarnames:
            #__dict__[key] dictionary used to store an object's attributes
            path.__dict__[varname]=np.nan*np.ones(par.Tpath) # create vector with shape (500, )      
            
    def find_steady_state(self,KY_ss,do_print=True): # notice default arguments are placed last
        """ find steady state """
        # unpacks namespace
        par = self.par
        
        # Create attribute by assigning a value 
        ss = self.ss

        # a. find A
        ss.K = KY_ss 
        # ignoring 2 last values (i.e. the factor prices: rk,w)
        Y,_,_ = production(par,1.0,ss.K) # <-- calling production function from outside RamseyModelClass with parameters, A=1, ss.K
        ss.A = 1/Y # <-- normalizing A

        # b. factor prices
        ss.Y,ss.rk,ss.w = production(par,ss.A,ss.K) # we unpack the solutions
        # assert statement is used to continue the execute if the given condition np.isclose() evaluates to True.
        assert np.isclose(ss.Y,1.0) # Returns a boolean array where two arrays are element-wise equal within a tolerance. compares ss.Y and 1.0
        
        # a condition for asset market to clear
        ss.r = ss.rk-par.delta

        # c. implied discount factor in steady state (which follows from euler equation!)
        par.beta = 1/(1+ss.r)

        # d. aggregated consumption (from good markets it follows that C=Y-I)
        ss.C = ss.Y - par.delta*ss.K

        if do_print:
            # f-string formatted printout with fixed 4 digits after decimal
            print(f'Y_ss = {ss.Y:.4f}')
            print(f'K_ss/Y_ss = {ss.K/ss.Y:.4f}')
            print(f'rk_ss = {ss.rk:.4f}')
            print(f'r_ss = {ss.r:.4f}')
            print(f'w_ss = {ss.w:.4f}')
            print(f'A = {ss.A:.4f}')
            print(f'beta = {par.beta:.4f}')

    def evaluate_path_errors(self):
        """ evaluate errors along transition path. """
        # unpacks namespaces
        par = self.par
        ss = self.ss
        path = self.path

        # a. consumption        
        C = path.C
        C_plus = np.append(path.C[1:],ss.C) # <-- append steady state solutions for C to our array path C as the last value

        # b. capital
        K = path.K
        K_lag = path.K_lag = np.insert(K[:-1],0,par.K_lag_ini) # <-- insert K_lag_ini as element 0 in array K
        
        # c. production and factor prices
        path.Y,path.rk,path.w = production(par,path.A,K_lag) # unpack solutions
        path.r = path.rk-par.delta
        
        r_plus = np.append(path.r[1:],ss.r) # <-- append steady state solutions for r to our array r_plus as the last value

        # d. errors (also called H)
        errors = np.nan*np.ones((2,par.Tpath)) # (2, 500) dimensional
        errors[0,:] = C**(-par.sigma) - par.beta*(1+r_plus)*C_plus**(-par.sigma) # <-- insert error value for Euler equation as row zero in array
        errors[1,:] = K - ((1-par.delta)*K_lag + (path.Y - C)) # <-- <-- insert error in law of motion of capital as row one in array
        
        # return whole vector of errors
        return errors.ravel() # A 1-D array with shape (1000,) with input values from above errors[0,:], erros[1,:]
    
    def calculate_jacobian(self,h=1e-6):
        # The jacobian is just a gradient. I.e. the matrix of what the implied errors are in H when a single K or C change.
        """ calculate jacobian. """

        # unpacks namespaces
        par = self.par
        ss = self.ss
        path = self.path
        
        # a. allocate (how big is the Jacobian)
        Njac = 2*par.Tpath # <-- 2 equations in each period: 2*500=1000
        jac = self.jac = np.nan*np.ones((Njac,Njac)) # (1000,1000)
        
        # built x that has same size as the jacobian
        x_ss = np.nan*np.ones((2,par.Tpath)) # (2, 500)
        x_ss[0,:] = ss.C
        x_ss[1,:] = ss.K
        
        # turn into vector
        x_ss = x_ss.ravel() # <-- combine ss.C as row 1 and ss.K as row 2

        # b. baseline errors (errors in ss and should be close to zero!)
        path.C[:] = ss.C
        path.K[:] = ss.K
        base = self.evaluate_path_errors() # <-- prepend with self as we call method within class

        # c. calculate the jacobian!
        for i in range(Njac): # <-- loop through each variable in Njac
            
            # i. add small number h to a single x (single K or C) 
            x_jac = x_ss.copy()
            x_jac[i] += h # <-- increment with small number h
            x_jac = x_jac.reshape((2,par.Tpath)) # <-- reshape into matrix
            
            # ii. alternative errors
            path.C[:] = x_jac[0,:]
            path.K[:] = x_jac[1,:]
            alt = self.evaluate_path_errors() # evaluate the alternative errors

            # iii. numerical derivative: [f(x+h)-f(x)]/h
            jac[:,i] = (alt-base)/h # <-- doing it for the i'th column of the jacobian
        
    def solve(self,do_print=True):
        """ solve for the transition path """

        # unpack namespace
        par = self.par
        ss = self.ss
        path = self.path
        
        # a. define equation system
        def eq_sys(x):
            
            # i. update
            x = x.reshape((2,par.Tpath))
            path.C[:] = x[0,:]
            path.K[:] = x[1,:]
            
            # ii. return errors
            return self.evaluate_path_errors()

        # b. initial guess that our x's are the ss
        x0 = np.nan*np.ones((2,par.Tpath))
        x0[0,:] = ss.C
        x0[1,:] = ss.K
        x0 = x0.ravel() # <-- turn into vector

        # c. call solver
        if par.solver == 'broyden':

            x = broyden_solver(eq_sys,x0,self.jac,do_print=do_print)
        
        elif par.solver == 'scipy':
            
            root = optimize.root(eq_sys,x0,method='hybr',options={'factor':1.0})
            # the factor determines the size of the initial step
            #  too low: slow
            #  too high: prone to errors
            
            x = root.x # <-- return

        else:

            raise NotImplementedError('unknown solver')
            

        # d. final evaluation
        eq_sys(x)


# defining our production function based on general (CES) or special type (cobb-douglas)
def production(par,A,K_lag):
    """ production and factor prices """

    # a. production and factor prices
    if par.production_function == 'ces':

        # a. production
        Y = A*( par.alpha*K_lag**(-par.theta) + (1-par.alpha)*(1.0)**(-par.theta) )**(-1.0/par.theta)

        # b. factor prices
        rk = A*par.alpha*K_lag**(-par.theta-1) * (Y/A)**(1.0+par.theta)
        w = A*(1-par.alpha)*(1.0)**(-par.theta-1) * (Y/A)**(1.0+par.theta)

    elif par.production_function == 'cobb-douglas':

        # a. production
        Y = A*K_lag**par.alpha * (1.0)**(1-par.alpha)

        # b. factor prices
        rk = A*par.alpha * K_lag**(par.alpha-1) * (1.0)**(1-par.alpha)
        w = A*(1-par.alpha) * K_lag**(par.alpha) * (1.0)**(-par.alpha)

    else:
    # if we input other type we shall raise an error message with the following text
        raise Exception('unknown type of production function')
    
    # return production and factor prices
    return Y,rk,w            


def broyden_solver(f,x0,jac,tol=1e-8,maxiter=100,do_print=False):
    """ numerical equation system solver using the broyden method 
    
        f (callable): function return errors in equation system
        jac (ndarray): initial jacobian
        tol (float,optional): tolerance
        maxiter (int,optional): maximum number of iterations
        do_print (bool,optional): print progress

    """

    # a. initial
    x = x0.ravel() # <-- takes input
    y = f(x) # <-- evaluates function

    # b. iterate
    for it in range(maxiter):
        
        # i. current difference
        abs_diff = np.max(np.abs(y)) # <-- how many non-zeros do we have
        if do_print: print(f' it = {it:3d} -> max. abs. error = {abs_diff:12.8f}')
        
        if abs_diff < tol:
            return x # <-- if errors zero then stop
        
        # ii. new x (updating step)
        dx = np.linalg.solve(jac,-y)
        assert not np.any(np.isnan(dx))
    
        # iii. evaluate (updating the Jacobian until error is small)
        ynew = f(x+dx)
        dy = ynew-y
        jac = jac + np.outer(((dy - jac @ dx) / np.linalg.norm(dx)**2), dx) 
        y = ynew
        x += dx
            
    else:

        raise ValueError(f'no convergence after {maxiter} iterations') # <-- if it takes too long raise error