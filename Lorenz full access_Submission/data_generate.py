import os
print(os.getcwd())

import numpy as np

def iter_rk45(prev, t, h, f, fargs=None): # on wikipedia this is simply called the RK4 method.
    
    if fargs == None:
        z1 = prev
        z2 = prev + (h/2)*f(t, z1)
        z3 = prev + (h/2)*f(t + 0.5*h, z2)
        z4 = prev + h*f(t + 0.5*h, z3)

        z = (h/6)*(f(t, z1) + 2*f(t + 0.5*h, z2) + 2*f(t + 0.5*h, z3) + f(t + h, z4))
        curr = prev + z
    
    else:
        z1 = prev
        z2 = prev + (h/2)*f(t, z1, fargs)
        z3 = prev + (h/2)*f(t + 0.5*h, z2, fargs)
        z4 = prev + h*f(t + 0.5*h, z3, fargs)

        z = (h/6)*(f(t, z1, fargs) + 2*f(t + 0.5*h, z2, fargs) + 2*f(t + 0.5*h, z3, fargs) + f(t + h, z4, fargs))
        curr = prev + z
    
    return curr

def rk45(f, t_span, sol_init, h, fargs=None):
    start = t_span[0]
    end = t_span[1]
    
    t_eval = np.arange(start, end+h, h)
    sol_len = len(t_eval)
    solution = [0] * len(t_eval)
    
    solution[0] = sol_init
    prev = sol_init
    
    for t_id in range(1, sol_len):
        t = t_eval[t_id-1] # there was an index error here, which I have now corrected. Formerly it was t_id not t_id - 1. Corrected on 13 June 2024, Thursday.
        curr = iter_rk45(prev, t, h, f, fargs)
        solution[t_id] = curr
        prev = curr
    
    return t_eval, np.array(solution)


# Function to generate initial conditions

def gen_init_cond(length, center, sd, f, h, fargs=None, t_stabilise=0, pdf="gaussian", seed = None, run_till_on_attractor = False):
    """
    Generates an array of length of length of random initial conditions centered at center, with standard deviation sd.
    If run_till_on_attractor == True then it runs the dynamical system using rk45 for time 0 to t_stabilise.
    """
    np.random.seed(seed)

    z = np.array([center for i in range(length)])

    if pdf == "gaussian":
        z += np.random.normal(0, sd, (length, len(center)))
        
    elif pdf == "uniform":
        z += np.random.uniform(-sd, sd, (length, len(center)))

    else:
        print('No such pdf')

    if run_till_on_attractor:    
        for i in range(len(z)):
            z[i] = rk45(f, (0,t_stabilise), z[i], h, fargs)[1][-1]
    
    return z
    