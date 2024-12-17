#%%
import numpy as np

from datagen.data_generate import rk45
from datagen.data_generate import gen_init_cond
from estimators.esn_funcs import ESN
from utils.crossvalidation import CrossValidate
from time import time
from utils.normalisation import normalise_arrays
from estimators import ESN_helperfunctions as esn_help

# Create the Rossler dataset
def rossler(t, Z, args):
    u, v, w = Z
    a, b, c = args
    
    up = - v - w
    vp = u + a * v
    wp = b + w * (u - c)
    return np.array([up, vp, wp])

ros_args = (1/10, 1/10, 14)
Z0 = (2.0, 1.0, 5.0)
h = 0.005
t_span = (0, 250)
slicing = int(h/h)
t_trans = 10
Z0 = gen_init_cond(1, Z0, 0, rossler, 2*t_trans, h,
                    ros_args, pdf = 'gaussian', run_till_on_attractor = True)[0]
print('Z0', Z0)

t_eval, data = rk45(rossler, t_span, Z0, h, ros_args)
data = 0.01 * data
t_eval = t_eval[::slicing]
data = data[::slicing]

# Define full data training and testing sizes
ndata  = len(data)
ntrain = 60000 
washout = 2000
ntest = ndata - ntrain
N = 1000
d_in, d_out = 3, 3
ld = 1e-13
gamma = 3.7
spec_rad = 1.2
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
ld_range = np.logspace(-16,-12,25) #np.logspace(-14,-9,3)  
gamma_range = np.linspace(3,4.9,40) # np.linspace(2,5,3) 
spec_rad_range = np.linspace(0,1.2,13) # np.linspace(0.25, 2, 3) 
s_range = np.array([0]) # np.linspace(0, 1, 3)
param_ranges = [ld_range, gamma_range, spec_rad_range, s_range]

"""
ld_range =  [ld, ld+1] 
gamma_range =   [gamma, gamma+1] #
spec_rad_range =   [spec_rad, spec_rad+0.3] #
s_range = [s, 1] # 
param_ranges = [ld_range, gamma_range, spec_rad_range, s_range]
"""
"""
ld_range = [ld]
gamma_range = [gamma]
spec_rad_range = [spec_rad]
s_range = [s]
param_ranges = [ld_range, gamma_range, spec_rad_range, s_range]
"""

# Define the names of the parameters -- orders must match
param_names = ["ld", "gamma", "spec rad", "s"]
# Define the additional inputs taken in by the 
param_add = [N, d_in, d_out, washout]

#%%
print(train_in_esn.shape, train_teach_esn.shape)
print(param_ranges, param_add)
print(len(ld_range) * len(gamma_range) * len(spec_rad_range) * len(s_range) * 8 /3600)
print(len(ld_range))
print(len(gamma_range))
print(len(spec_rad_range))
print(len(s_range))

#%%
train_size = 20000
validation_size = 20000
nstarts = 5

t1 = time()
if __name__ == "__main__":
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

best_params1 = {'validation_error': 434.167717435026, 'ld': 1e-09, 'gamma': 5.0, 'spec rad': 0.25, 's': 1}

#%%
import matplotlib.pyplot as plt
plt.plot(train_in_esn[:20000,-1:].reshape((20000,)))
plt.show()
print(np.min(train_in_esn[:20000]))
#%%
print(np.mean(train_in_esn[:20000]))

#%%
esn = ESN(ld,gamma, spec_rad,s,N,d_in,d_out,washout)
esn = esn.Train(train_in_esn[:20000], train_teach_esn[:20000])
print(esn.W.shape)
print(esn.bias.shape)
print(esn.x_start_forecasting.shape)
print(esn.x_start_path_continue.shape)
p = esn.PathContinue(train_teach_esn[:20000][-1], 2000)
#%%
print((esn.A == np.zeros((N,N))).all == True)
print(np.max(esn.A))
print(np.mean(esn.A))
print(np.max(esn.C))
print(np.mean(esn.C))
print(np.sum(esn.zeta))
#%%
print(esn.x_start_path_continue)
print(np.mean(esn.x_start_path_continue))
#%%
targets = np.transpose(train_teach_esn[:20000])
print(targets[-1])
print(np.mean(targets))
#%%
print(np.max(esn.W))
print(np.mean(esn.W))
print(np.max(esn.bias))
print(np.mean(esn.bias))

#%%
print(train_teach_esn[:20000][-1])
print(np.mean(esn.x_start_path_continue))
print(p[1000])
#%%
print(p.shape)
#print(p)

import matplotlib.pyplot as plt
plt.plot(p[:,0,0])

#%% Best parameters from run 1 11 March 2024, Monday
"""
Grid




# Define the range of parameters for which you want to cross validate over
ld_range = np.logspace(-15,-8,15) #np.logspace(-14,-9,3)  
gamma_range = np.linspace(2.5,5.5,16) # np.linspace(2,5,3) 
spec_rad_range = np.linspace(0.6,2.0, 8) # np.linspace(0.25, 2, 3) 
s_range = np.array([0]) # np.linspace(0, 1, 3)
param_ranges = [ld_range, gamma_range, spec_rad_range, s_range]

Intermediary Best Parameters:
Validation errors: 0.0009204835153581827
ld: 3.1622776601683794e-15
gamma: 2.9
spec rad: 0.6
s: 0

"""
#%% Best parameters from run 2 12 March 2024, Tuesday

"""
Grid

# Define the range of parameters for which you want to cross validate over
ld_range = np.logspace(-17,-8,19) #np.logspace(-14,-9,3)  
gamma_range = np.linspace(2.1,5.5,13) # np.linspace(2,5,3) 
spec_rad_range = np.linspace(0,2.0,11) # np.linspace(0.25, 2, 3) 
s_range = np.array([0]) # np.linspace(0, 1, 3)
param_ranges = [ld_range, gamma_range, spec_rad_range, s_range]

----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.00023988325735082222
ld: 1e-13
gamma: 4.933333333333334
spec rad: 0.6000000000000001
s: 0
----------------------------------------
"""

#%% Best parameters from run 3 14 March 2024, Thursday

"""
Grid


# Define the range of parameters for which you want to cross validate over
ld_range = np.logspace(-17,-8,19) #np.logspace(-14,-9,3)  
gamma_range = np.linspace(2.1,5.5,13) # np.linspace(2,5,3) 
spec_rad_range = np.linspace(0,2.0,11) # np.linspace(0.25, 2, 3) 
s_range = np.array([0]) # np.linspace(0, 1, 3)
param_ranges = [ld_range, gamma_range, spec_rad_range, s_range]

----------------------------------------
Intermediary Best Parameters:
Validation errors: 4.4368130849843415e-07
ld: 3.1622776601683794e-15
gamma: 4.65
spec rad: 0.2
s: 0
"""

#%% Best parameters from run 4 5 April 2024, Friday
"""
----------------------------------------
Intermediary Best Parameters:
Validation errors: 8.472058639275968e-05
ld: 6.812920690579594e-14
gamma: 4.51025641025641
spec rad: 0.39999999999999997
s: 0

Grid 

ld_range = np.logspace(-16,-12,25) #np.logspace(-14,-9,3)  
gamma_range = np.linspace(3,4.9,40) # np.linspace(2,5,3) 
spec_rad_range = np.linspace(0,1.2,13) # np.linspace(0.25, 2, 3) 
s_range = np.array([0]) # np.linspace(0, 1, 3)
param_ranges = [ld_range, gamma_range, spec_rad_range, s_range]


Intermediary Best Parameters:
Validation errors: 3986749.2106039515
ld: 1e-16
gamma: 3.0
spec rad: 0.0
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 137.55982022071098
ld: 1e-16
gamma: 3.0
spec rad: 0.09999999999999999
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 74.07757960911718
ld: 1e-16
gamma: 3.0
spec rad: 0.3
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.11437377882180177
ld: 1e-16
gamma: 3.0
spec rad: 1.2
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.07896489129575394
ld: 1.4677992676220675e-16
gamma: 3.0
spec rad: 1.2
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.004279204813841276
ld: 2.1544346900318868e-16
gamma: 3.5358974358974358
spec rad: 0.0
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.003763711626164808
ld: 1.4677992676220676e-15
gamma: 3.0
spec rad: 0.0
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.0008715431687401521
ld: 1.4677992676220676e-15
gamma: 3.0
spec rad: 0.09999999999999999
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.0006357215499070629
ld: 2.1544346900318867e-15
gamma: 3.0487179487179485
spec rad: 0.09999999999999999
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.0005631379684050168
ld: 3.1622776601683794e-15
gamma: 3.146153846153846
spec rad: 0.09999999999999999
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.00015940896831515654
ld: 3.1622776601683794e-15
gamma: 3.292307692307692
spec rad: 0.39999999999999997
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.00011106047376839256
ld: 2.1544346900318778e-14
gamma: 4.9
spec rad: 0.49999999999999994
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 8.472058639275968e-05
ld: 6.812920690579594e-14
gamma: 4.51025641025641
spec rad: 0.39999999999999997
s: 0

"""

#%% Cross validation parameters found from cross validation run on Tue 23 April 2024


"""
Best parameters

ld: 4.641588833612772e-15
gamma: 3.2435897435897436
spec rad: 0.39999999999999997
s: 0

Grid

# Define the range of parameters for which you want to cross validate over
ld_range = np.logspace(-16,-12,25) #np.logspace(-14,-9,3)  
gamma_range = np.linspace(3,4.9,40) # np.linspace(2,5,3) 
spec_rad_range = np.linspace(0,1.2,13) # np.linspace(0.25, 2, 3) 
s_range = np.array([0]) # np.linspace(0, 1, 3)
param_ranges = [ld_range, gamma_range, spec_rad_range, s_range]


done cross validating
Intermediary Best Parameters:
Validation errors: 16508.60581761149
ld: 1e-16
gamma: 3.0
spec rad: 0.0
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 4523.460306097557
ld: 1e-16
gamma: 3.0
spec rad: 0.09999999999999999
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 539.52701134879
ld: 1e-16
gamma: 3.0
spec rad: 0.19999999999999998
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 54.85726521899336
ld: 1e-16
gamma: 3.0
spec rad: 1.0999999999999999
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 28.907933268042076
ld: 1e-16
gamma: 3.0
spec rad: 1.2
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 8.74366647287151
ld: 1e-16
gamma: 3.2435897435897436
spec rad: 1.2
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 5.343364515368862
ld: 1e-16
gamma: 3.4384615384615387
spec rad: 1.2
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.01745208717538383
ld: 1.4677992676220675e-16
gamma: 3.0487179487179485
spec rad: 0.0
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.013458062952230529
ld: 6.812920690579622e-16
gamma: 4.364102564102565
spec rad: 0.0
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.0018301901118187307
ld: 1e-15
gamma: 4.461538461538462
spec rad: 0.0
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.001132505601266089
ld: 1.4677992676220676e-15
gamma: 3.0
spec rad: 0.09999999999999999
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.00068926165039233
ld: 1.4677992676220676e-15
gamma: 3.0
spec rad: 0.19999999999999998
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.0006828102250606486
ld: 2.1544346900318867e-15
gamma: 3.0974358974358975
spec rad: 0.3
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.0005517035352508762
ld: 2.1544346900318867e-15
gamma: 3.146153846153846
spec rad: 0.09999999999999999
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.0004713712564728124
ld: 2.1544346900318867e-15
gamma: 3.2435897435897436
spec rad: 0.09999999999999999
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.00027231637724659235
ld: 2.1544346900318867e-15
gamma: 3.341025641025641
spec rad: 0.39999999999999997
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.00023583991218266613
ld: 3.1622776601683794e-15
gamma: 3.0487179487179485
spec rad: 0.39999999999999997
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.0001359871886611024
ld: 3.1622776601683794e-15
gamma: 3.2435897435897436
spec rad: 0.39999999999999997
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.00012786067896796007
ld: 3.1622776601683794e-15
gamma: 3.8282051282051284
spec rad: 0.3
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 7.41296261242564e-05
ld: 3.1622776601683794e-15
gamma: 4.0717948717948715
spec rad: 0.39999999999999997
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 5.787461889021944e-05
ld: 4.641588833612772e-15
gamma: 3.2435897435897436
spec rad: 0.39999999999999997
s: 0
"""
