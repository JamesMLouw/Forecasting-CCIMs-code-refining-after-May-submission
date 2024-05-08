# sample code 
 
import cmath
import numpy as np
import scipy
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 25

def log(x):
    # helper function apply complex log to array element-wise
    # args: x - an array of complex or real scalars. 
    return np.array([cmath.log(xx) for xx in x])

def iter_rk45(prev, t, h, f, fargs=None):
    # helper function one iteration of rk45
    # args: prev - output from previous time step. 
          # t - time, if required in f. If not used in f, any value will do.
          # h - the size of time step used to generate next data point
          # f - callable with format def f(t, inputs, additional args)
              # additional args is default None. 
    
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

# generate full lyapunov spectrum
def lyapunov_spectrum(data, N, h, eq_type, t_trans, jacobian, delta=10**(-7), seed=None, lyap_store=3, gram_schmidt = 15, t_cap = 10000):
    # compute lyapunov spectrum from data and known jacobian function 
    # Uses Bennetin's algorithm (see Appendix A of https://link.springer.com/book/10.1007/978-3-642-14938-2)
    # args: data - can be single dimensional or 2-dimensional but data should 
                 # be stored row-wise so that data[t] is the t-th data point.
          # N - dimension of system and also the number of exponents found
          # h - time step size that was used to generate dataset
          # eq_type - either differential (e.g. for lorenz, mackey-glass) 
                    # or difference (e.g. for esn)
          # t_trans - initial transient period before summing for final average
          # jacobian - callable that takes in (t, inputs)
          # delta - size of initial perturbation vector. Should be small.                
          # lyap_store - number of lyapunov exponents to plot
          # gram_schmidt - number of iterations before reorthogonalizing vectors
          # t_cap - cap for number of time steps used to compute lyapunov spectrum
    
    if seed != None:
        np.random.seed(seed)
        Delta = delta * scipy.linalg.orth(np.random.rand(N, N)) 
    else:
        Delta = delta * np.identity(N)
    
    vec_len_sum = np.zeros(shape=(N, ), dtype='float32')
    
    t_end = min(len(data), t_cap)
    lyap_store = min(lyap_store, N)
    lyapunov_spec_time = np.zeros(shape = (t_end//gram_schmidt + 1, lyap_store))
    
    for t in range(t_end):
        data_t = data[t]
        
        if t == t_trans:
            Delta = np.linalg.qr(Delta)[0]
            
        if t % gram_schmidt == 0:
            if t <= t_trans:
                total_time = (t + 1) * h
            else:
                total_time = (t - t_trans + 1) * h
                
            lyapunov_spec = np.real(vec_len_sum) / total_time
            print(t, ": ", lyapunov_spec[0:lyap_store])
            lyapunov_spec_time[t//gram_schmidt,:] = np.flip(np.sort(lyapunov_spec))[0:lyap_store]

        if eq_type == "ordinary":
            for j in range(N):
                delta_j = lambda t, dy : jacobian(t, data_t) @ dy
                Delta[:, j] = iter_rk45(Delta[:, j], t, h, delta_j)
                
        if eq_type == "difference":
            Delta = jacobian(t, data_t) @ Delta
            
        if t % gram_schmidt == 0:
            Delta, R = np.linalg.qr(Delta)
            evolved_vec_lengths = np.diagonal(R)
            sizes = np.absolute(evolved_vec_lengths)
            print(t, ': ', min(sizes), max(sizes))
    
            if t >= t_trans:
                vec_len_sum = vec_len_sum + log(evolved_vec_lengths)
        
    Delta, R = np.linalg.qr(Delta)
    evolved_vec_lengths = np.diagonal(R) 
    vec_len_sum = vec_len_sum + log(evolved_vec_lengths)
  
    total_time = (t_end-t_trans) * h
    lyapunov_spec = np.real(vec_len_sum) / total_time
    
    lyap_exp_plot, (lyap_exp_ax, bound_ax) = plt.subplots(2, 1, figsize = (20,5))
    time_steps = [i*gram_schmidt for i in range(t_end//gram_schmidt + 1)]
    lyapunov_spec_bound = [t*l for (t,l) in zip(time_steps, lyapunov_spec_time[:,0])]
    
    for i in range(lyap_store):
        lyap_exp_ax.plot(time_steps[2*t_trans//gram_schmidt:], lyapunov_spec_time[2*t_trans//gram_schmidt:,i], label = '\lambda'+str(i))

    bound_ax.plot(time_steps[2*t_trans//gram_schmidt:], lyapunov_spec_bound[2*t_trans//gram_schmidt:])
        
    lyap_exp_ax.set_title('Convergence of Lyapunov exponents')
    lyap_exp_ax.set_xlabel('time')
    lyap_exp_ax.set_ylabel('Lyapunov exponents')
    plt.legend()
    lyap_exp_plot.savefig('lyapunov_exponents_convergence.pdf')

    
    return lyapunov_spec, lyapunov_spec_time

# generate top lyapunov exponent
def top_lyap_exp(data, N, h, eq_type, t_trans, jacobian, delta=10**(-7), seed=None, normalise = 15, t_cap = 10000):
    # compute lyapunov spectrum from data and known jacobian function 
    # Uses Bennetin's algorithm (see Appendix A of https://link.springer.com/book/10.1007/978-3-642-14938-2)
    # args: data - can be single dimensional or 2-dimensional but data should 
                 # be stored row-wise so that data[t] is the t-th data point.
          # N - dimension of system and also the number of exponents found
          # h - time step size that was used to generate dataset
          # eq_type - either differential (e.g. for lorenz, mackey-glass) 
                    # or difference (e.g. for esn)
          # t_trans - initial transient period before summing for final average
          # jacobian - callable that takes in (t, inputs)
          # delta - size of initial perturbation vector. Should be small.                
          # normalise - number of iterations before normalising vector
          # t_cap - cap for number of time steps used to compute lyapunov spectrum
    
    np.random.seed(seed)
    v = np.random.rand(N,1)
    Delta = delta / np.linalg.norm(v) * v
    
    vec_len_sum = 0
    
    t_end = min(len(data), t_cap)
    lyapunov_exp_time = np.zeros(shape = (t_end//normalise + 1, ))
        
    for t in range(t_end):
        data_t = data[t]
        
        if t == t_trans:    
            evolved_vec_length = np.linalg.norm(Delta)
            Delta = 1/evolved_vec_length * Delta
        
        if t % normalise == 0:
            if t <= t_trans:
                total_time = (t + 1) * h
            else:
                total_time = (t - t_trans) * h
                
            lyapunov_exp = np.real(vec_len_sum) / total_time
            print(t, ": ", lyapunov_exp)
            lyapunov_exp_time[t//normalise] = lyapunov_exp

        if eq_type == "ordinary":
            delta_0 = lambda t, dy : jacobian(t, data_t) @ dy
            Delta[:, 0] = iter_rk45(Delta[:, 0], t, h, delta_0)
                
        if eq_type == "difference":
            Delta = jacobian(t, data_t) @ Delta
            
        if t % normalise == 0:
            evolved_vec_length = np.linalg.norm(Delta)
            Delta = 1/evolved_vec_length * Delta
            
            if t >= t_trans:
                vec_len_sum = vec_len_sum + np.log(evolved_vec_length)
        
    evolved_vec_length = np.linalg.norm(Delta)
    vec_len_sum = vec_len_sum + np.log(evolved_vec_length)
  
    total_time = (t_end-t_trans) * h
    lyapunov_exp = np.real(vec_len_sum) / total_time

    
    lyap_exp_plot, (lyap_exp_ax, bound_ax) = plt.subplots(2, 1, figsize = (20,5))
    time_steps = [i*normalise for i in range(t_end//normalise + 1)]
    lyapunov_exp_bound = [t*l for (t,l) in zip(time_steps, lyapunov_exp_time[:,0])]
    
    lyap_exp_ax.plot(time_steps[2*t_trans//normalise:], lyapunov_exp_time[2*t_trans//normalise:], label = '\lambda_1')
    bound_ax.plot(time_steps[2*t_trans//normalise:], lyapunov_exp_bound[2*t_trans//normalise:])
        
    lyap_exp_ax.set_title('Convergence of top Lyapunov exponent')
    lyap_exp_ax.set_xlabel('time')
    lyap_exp_ax.set_ylabel('Top Lyapunov exponent')
    bound_ax.set_ylabel('Log error bound')

    plt.legend()
    lyap_exp_plot.savefig('top_lyapunov_exponent_convergence.pdf')

    
    return lyapunov_exp, lyapunov_exp_time
    

    
def kaplan_yorke(sorted_array):
    
    exp_sum = 0
    D_KY = 0

    for idx in range(0, len(sorted_array)):
        exp_sum = exp_sum + sorted_array[idx]
        if exp_sum < 0:
            D_KY = idx - 1 - ((exp_sum - sorted_array[idx])/sorted_array[idx])
            break
            
    D_KY = D_KY + 1
    
    return D_KY