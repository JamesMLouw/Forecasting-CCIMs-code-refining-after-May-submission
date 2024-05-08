#%%
import numpy as np

from datagen.data_generate import rk45
from estimators.esn_funcs import ESN
from utils.crossvalidation import CrossValidate
from time import time
from utils.normalisation import normalise_arrays
from estimators import ESN_helperfunctions as esn_help

# Create the Lorenz dataset
def lorenz(t, Z, args):
    u, v, w = Z
    sig, beta, rho = args
    
    up = -sig*(u - v)
    vp = rho*u - v - u*w
    wp = -beta*w + u*v
    
    return np.array([up, vp, wp])

lor_args = (10, 8/3, 28)
Z0 = (0, 1, 1.05)

h = 0.005
t_span = (0, 100)
slicing = int(h/h)

t_eval, data = rk45(lorenz, t_span, Z0, h, lor_args)
data = 0.01 * data
t_eval = t_eval[::slicing]
data = data[::slicing]

# Define full data training and testing sizes
ndata  = len(data)
ntrain = 20000 
washout = 1000
ntest = ndata - ntrain
N = 1000
d_in, d_out = 3, 3
ld = 10**(-13) 
gamma = 3.7
spec_rad = 1.2
norm_A = 0.1
s = 0


# Construct training input and teacher, testing input and teacher
train_in_esn = data[0:ntrain] 
train_teach_esn = data[0:ntrain]

"""
# Normalise training arrays if necessary
normalisation_output = normalise_arrays([training_input_orig, training_teacher_orig], norm_type=None)
train_in_esn, train_teach_esn = normalisation_output[0]
shift_esn, scale_esn = normalisation_output[1], normalisation_output[2]
"""

# Define the range of parameters for which you want to cross validate over

ld_range = np.logspace(-17,-9,17) # np.logspace(-17, -7, 41)  # [ld, ld+1] 
gamma_range =  np.linspace(2,6,41) # np.linspace(2, 5, 31) # [gamma, gamma+1] #
spec_rad_range =  np.linspace(0, 1.5, 16) # np.linspace(0.1, 3, 30) # [spec_rad, spec_rad+0.3] #
s_range = np.array([0]) # np.linspace(0, 1, 5) # [s, 1] # 
param_ranges = [ld_range, gamma_range, spec_rad_range, s_range]

"""
ld_range =  [ld, ld+1] 
gamma_range =   [gamma, gamma+1] #
spec_rad_range =   [spec_rad, spec_rad+0.3] #
s_range = [s, 1] # 
param_ranges = [ld_range, gamma_range, spec_rad_range, s_range]
"""
"""
ld_range =  [ld] 
gamma_range =   [gamma] #
spec_rad_range =   [spec_rad] #
s_range = [s] # 
param_ranges = [ld_range, gamma_range, spec_rad_range, s_range]
"""

# Define the names of the parameters -- orders must match
param_names = ["ld", "gamma", "spec rad", "s"]
# Define the additional inputs taken in by the 
param_add = [N, d_in, d_out, washout]

print(train_in_esn.shape, train_teach_esn.shape)
print(param_ranges, param_add)

#%%
train_size = 12000
validation_size = 4000
nstarts = 5

if __name__ == "__main__":
    
    t1 = time()
    CV = CrossValidate(validation_parameters=[train_size, validation_size, nstarts],
                       validation_type="rolling", task="PathContinue", norm_type=None)
    print('created CV')
    best_parameters, parameter_combinations, errors = CV.crossvalidate_multiprocessing(ESN,
                                        train_in_esn, train_teach_esn, param_ranges, 
                                        param_names, param_add, num_processes=25)
    t2 = time()
print(t2-t1)
#%%
print(best_parameters)

#%%
# for lorenz with full access

best_params1 = {'validation_error': 434.167717435026, 'ld': 1e-09, 'gamma': 5.0, 'spec rad': 0.25, 's': 1}

#%%

#%%

esn = ESN(ld,gamma, spec_rad,s,N,d_in,d_out,washout)
esn = esn.Train(train_in_esn, train_teach_esn)
p = esn.PathContinue(train_teach_esn[-1], 1000)
print(p.shape)

# %%
"""
Cross validation run 15 March 2024, Fri
Grid

# Define the range of parameters for which you want to cross validate over

ld_range = np.logspace(-17,-9,17) # np.logspace(-17, -7, 41)  # [ld, ld+1] 
gamma_range =  np.linspace(2,6,21) # np.linspace(2, 5, 31) # [gamma, gamma+1] #
spec_rad_range =  np.linspace(0, 2, 21) # np.linspace(0.1, 3, 30) # [spec_rad, spec_rad+0.3] #
s_range = np.array([0]) # np.linspace(0, 1, 5) # [s, 1] # 
param_ranges = [ld_range, gamma_range, spec_rad_range, s_range]



Intermediary Best Parameters:
Validation errors: 206014319322.5856
ld: 1e-17
gamma: 2.0
spec rad: 0.0
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 472286.6297027236
ld: 1e-17
gamma: 2.0
spec rad: 0.1
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 113876.69493401815
ld: 1e-17
gamma: 2.0
spec rad: 0.2
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 2971.8599764033806
ld: 1e-17
gamma: 2.0
spec rad: 0.30000000000000004
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 761.816351769721
ld: 1e-17
gamma: 2.0
spec rad: 0.9
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 66.59783696814517
ld: 1e-17
gamma: 2.0
spec rad: 1.2000000000000002
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.17543046347975286
ld: 1e-17
gamma: 2.0
spec rad: 1.3
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.1487647262062604
ld: 1e-17
gamma: 2.0
spec rad: 1.5
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.11662095766132048
ld: 1e-17
gamma: 2.0
spec rad: 1.6
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.02644725745403053
ld: 1e-17
gamma: 2.2
spec rad: 1.3
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.023517925021336834
ld: 3.1622776601683794e-15
gamma: 2.0
spec rad: 0.5
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.023141732084306844
ld: 3.1622776601683794e-15
gamma: 2.2
spec rad: 1.2000000000000002
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.0218878620730313
ld: 3.1622776601683794e-15
gamma: 2.6
spec rad: 0.4
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.020661748430955883
ld: 3.1622776601683794e-15
gamma: 3.2
spec rad: 0.2
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.019115797298268792
ld: 3.1622776601683794e-15
gamma: 3.6
spec rad: 0.2
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.017252220637541293
ld: 3.1622776601683794e-15
gamma: 4.0
spec rad: 0.1
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.016857611547323337
ld: 3.1622776601683796e-14
gamma: 2.2
spec rad: 1.2000000000000002
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.0166326395011182
ld: 3.1622776601683796e-14
gamma: 5.2
spec rad: 0.2
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.016545965580319906
ld: 1e-13
gamma: 5.800000000000001
spec rad: 0.1
s: 0
----------------------------------------

"""

#%%
"""
Cross validation 2 April 2024, Tu


Validation errors: 0.015099357061007384
ld: 1e-13
gamma: 2.1
spec rad: 1.2000000000000002
s: 0

Grid


ld_range = np.logspace(-17,-9,17) # np.logspace(-17, -7, 41)  # [ld, ld+1] 
gamma_range =  np.linspace(2,6,41) # np.linspace(2, 5, 31) # [gamma, gamma+1] #
spec_rad_range =  np.linspace(0, 1.5, 16) # np.linspace(0.1, 3, 30) # [spec_rad, spec_rad+0.3] #
s_range = np.array([0]) # np.linspace(0, 1, 5) # [s, 1] # 
param_ranges = [ld_range, gamma_range, spec_rad_range, s_range]


Intermediary Best Parameters:
Validation errors: 206014319322.5856
ld: 1e-17
gamma: 2.0
spec rad: 0.0
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 472286.6297027236
ld: 1e-17
gamma: 2.0
spec rad: 0.1
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 113876.69493401815
ld: 1e-17
gamma: 2.0
spec rad: 0.2
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 2971.8599764033806
ld: 1e-17
gamma: 2.0
spec rad: 0.30000000000000004
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 761.816351769721
ld: 1e-17
gamma: 2.0
spec rad: 0.9
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 66.59783696814517
ld: 1e-17
gamma: 2.0
spec rad: 1.2000000000000002
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.17543046347975286
ld: 1e-17
gamma: 2.0
spec rad: 1.3
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.1487647262062604
ld: 1e-17
gamma: 2.0
spec rad: 1.5
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.049072902765408835
ld: 1e-17
gamma: 2.1
spec rad: 1.5
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.02644725745403053
ld: 1e-17
gamma: 2.2
spec rad: 1.3
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.02105893841081114
ld: 1e-17
gamma: 2.3
spec rad: 1.3
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.020561831882848665
ld: 3.1622776601683794e-15
gamma: 2.1
spec rad: 0.7000000000000001
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.019115797298268792
ld: 3.1622776601683794e-15
gamma: 3.6
spec rad: 0.2
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.018936613278506036
ld: 3.1622776601683794e-15
gamma: 3.9000000000000004
spec rad: 0.2
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.017252220637541293
ld: 3.1622776601683794e-15
gamma: 4.0
spec rad: 0.1
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.01600044707555994
ld: 1e-14
gamma: 5.300000000000001
spec rad: 0.1
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.015099357061007384
ld: 1e-13
gamma: 2.1
spec rad: 1.2000000000000002
s: 0
"""