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
ntrain = 24000 
washout = 3000
ntest = ndata - ntrain
N = 1000
d_in, d_out = 1, 1
ld = 10**(-13) 
gamma = 3.7
spec_rad = 1.2
norm_A = 0.1
s = 0

# Construct training input and teacher, testing input and teacher
train_in_esn = data[0:ntrain] 
train_teach_esn = data[0:ntrain]

def f(x):
    return x[0] #1/56 * (np.sqrt(1201) - 9) * x[0] + x[1] 

trainin2 = np.zeros( len(train_in_esn) )
trainteach2 = np.zeros( len(train_teach_esn) )
 
trainin2 = np.array([ np.array([f(x)]) for x in train_in_esn ])
trainteach2 = np.array([ np.array([f(x)]) for x in train_teach_esn ])
	
del train_in_esn
del train_teach_esn

train_in_esn, train_teach_esn = trainin2, trainteach2


"""
# Normalise training arrays if necessary
normalisation_output = normalise_arrays([training_input_orig, training_teacher_orig], norm_type=None)
train_in_esn, train_teach_esn = normalisation_output[0]
shift_esn, scale_esn = normalisation_output[1], normalisation_output[2]
"""

# Define the range of parameters for which you want to cross validate over

ld_range = np.logspace(-16,-12,17) #np.logspace(-14,-9,3)  
gamma_range = np.linspace(2,5,16) # np.linspace(2,5,3) 
spec_rad_range = np.linspace(0,2,21) # np.linspace(0.25, 2, 3) 
s_range = np.array([0]) # np.linspace(0, 1, 3)
param_ranges = [ld_range, gamma_range, spec_rad_range, s_range]

"""
ld_range = np.logspace(-17,-13,17) 
gamma_range = np.linspace(3.3,4.9,17) 
spec_rad_range = np.linspace(0.9,1.7, 9) # np.linspace(0.25, 2, 3) 
s_range = np.array([0]) # np.linspace(0, 1, 3)
param_ranges = [ld_range, gamma_range, spec_rad_range, s_range]
"""

"""
ld_range = np.logspace(-17,-7,21) #np.logspace(-14,-9,3)  
gamma_range = np.linspace(2,6,13) # np.linspace(2,5,3) 
spec_rad_range = np.linspace(0.4,2.6, 12) # np.linspace(0.25, 2, 3) 
s_range = np.array([0,1]) # np.linspace(0, 1, 3)
param_ranges = [ld_range, gamma_range, spec_rad_range, s_range]
"""

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
train_size = 12000
validation_size = 4000
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
plt.plot(train_in_esn[:10000].reshape((10000,)))
plt.show()
print(np.min(train_in_esn[:10000]))

#%%
esn = ESN(ld,gamma, spec_rad,s,N,d_in,d_out,washout)
esn = esn.Train(train_in_esn[:10000], train_teach_esn[:10000])
print(esn.W.shape)
print(esn.bias.shape)
print(esn.x_start_forecasting.shape)
print(esn.x_start_path_continue.shape)
p = esn.PathContinue(train_teach_esn[:10000][-1], 2000)
#%%
print((esn.A == np.zeros((N,N))).all == True)
print(np.max(esn.A))
print(np.max(esn.C))
print(np.sum(esn.zeta))
#%%
print(np.max(esn.W))
print(np.max(esn.bias))

#%%
print(train_teach_esn[:10000][-1])
print(np.mean(esn.x_start_path_continue))
print(p[1000])
#%%
print(p.shape)
#print(p)

import matplotlib.pyplot as plt
plt.plot(p[:,0,0])

#%% Best parameters from run 1
"""
Grid

# Define the range of parameters for which you want to cross validate over
ld_range = np.logspace(-17,-7,11) #np.logspace(-14,-9,3)  
gamma_range = np.linspace(1,7,13) # np.linspace(2,5,3) 
spec_rad_range = np.linspace(0.2,3, 15) # np.linspace(0.25, 2, 3) 
s_range = [0] #np.linspace(0, 1, 5)
param_ranges = [ld_range, gamma_range, spec_rad_range, s_range]

Intermediary Best Parameters:
Validation errors: 0.003848540090983668
ld: 1e-13
gamma: 4.5
spec rad: 1.0
s: 0
------
"""

"""
We, 6 Mar 2024
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.0015789663870262021
ld: 3.1622776601683794e-15
gamma: 4.333333333333333
spec rad: 1.2000000000000002
s: 0

Over:

# Define the range of parameters for which you want to cross validate over
ld_range = np.logspace(-17,-7,21) #np.logspace(-14,-9,3)  
gamma_range = np.linspace(2,6,13) # np.linspace(2,5,3) 
spec_rad_range = np.linspace(0.4,2.6, 12) # np.linspace(0.25, 2, 3) 
s_range = np.array([0,1]) # np.linspace(0, 1, 3)
param_ranges = [ld_range, gamma_range, spec_rad_range, s_range]
"""
"""
Wednesday, 13 March 2024
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.005757602687922329
ld: 3.1622776601683794e-15
gamma: 4.1
spec rad: 1.2
s: 0

Grid

# Define the range of parameters for which you want to cross validate over

ld_range = np.logspace(-17,-13,17) 
gamma_range = np.linspace(3.3,4.9,17) 
spec_rad_range = np.linspace(0.9,1.7, 9) # np.linspace(0.25, 2, 3) 
s_range = np.array([0]) # np.linspace(0, 1, 3)
param_ranges = [ld_range, gamma_range, spec_rad_range, s_range]

"""
#%%

"""
Cross validation Wed 3 April 2024

----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.005894987158308241
ld: 3.1622776601683794e-15
gamma: 4.333333333333333
spec rad: 1.2000000000000002
s: 0

Grid

ld_range = np.logspace(-16,-12,17) #np.logspace(-14,-9,3)  
gamma_range = np.linspace(3,5,13) # np.linspace(2,5,3) 
spec_rad_range = np.linspace(0,2,21) # np.linspace(0.25, 2, 3) 
s_range = np.array([0]) # np.linspace(0, 1, 3)
param_ranges = [ld_range, gamma_range, spec_rad_range, s_range]

Intermediary Best Parameters:
Validation errors: 0.014057091652644649
ld: 1e-16
gamma: 3.0
spec rad: 0.0
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.010743883046725985
ld: 1e-16
gamma: 3.0
spec rad: 1.4000000000000001
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.01072690294912857
ld: 1e-16
gamma: 3.1666666666666665
spec rad: 1.3
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.009102109018936575
ld: 1e-16
gamma: 3.3333333333333335
spec rad: 1.3
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.008813194781884017
ld: 1e-16
gamma: 3.5
spec rad: 1.3
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.008286343820224588
ld: 1e-16
gamma: 3.833333333333333
spec rad: 1.3
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.00801926913212507
ld: 1e-16
gamma: 4.5
spec rad: 1.3
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.0075651258142053965
ld: 3.1622776601683793e-16
gamma: 3.6666666666666665
spec rad: 1.3
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.006577418389208933
ld: 5.623413251903491e-16
gamma: 3.833333333333333
spec rad: 1.3
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.006328719541153441
ld: 3.1622776601683794e-15
gamma: 3.1666666666666665
spec rad: 1.2000000000000002
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.005894987158308241
ld: 3.1622776601683794e-15
gamma: 4.333333333333333
spec rad: 1.2000000000000002
s: 0
----------------------------------------
t2 = time()
print(t2-t1)
40448.88657474518
"""
#%%
"""
Cross validation 22 April 2024 Mon


ld: 3.1622776601683794e-15
gamma: 3.4000000000000004
spec rad: 1.0
s: 0

Grid 

# Define the range of parameters for which you want to cross validate over

ld_range = np.logspace(-16,-12,17) #np.logspace(-14,-9,3)  
gamma_range = np.linspace(2,5,16) # np.linspace(2,5,3) 
spec_rad_range = np.linspace(0,2,21) # np.linspace(0.25, 2, 3) 
s_range = np.array([0]) # np.linspace(0, 1, 3)
param_ranges = [ld_range, gamma_range, spec_rad_range, s_range]


Intermediary Best Parameters:
Validation errors: 0.014241072300559066
ld: 1e-16
gamma: 2.0
spec rad: 0.0
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.011196510951861801
ld: 1e-16
gamma: 2.0
spec rad: 1.2000000000000002
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.01066378188667366
ld: 1e-16
gamma: 2.2
spec rad: 1.2000000000000002
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.010395653487889867
ld: 1e-16
gamma: 2.2
spec rad: 1.3
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.010021810645048566
ld: 1e-16
gamma: 2.4
spec rad: 1.8
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.009434793911827357
ld: 1e-16
gamma: 3.4000000000000004
spec rad: 1.2000000000000002
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.009177891264171005
ld: 1e-16
gamma: 4.6
spec rad: 1.2000000000000002
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.008915901321936728
ld: 1.7782794100389228e-16
gamma: 4.4
spec rad: 1.2000000000000002
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.008869850959458123
ld: 3.1622776601683793e-16
gamma: 4.6
spec rad: 1.2000000000000002
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.00883858420464348
ld: 1e-15
gamma: 2.0
spec rad: 0.8
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.006483271318095915
ld: 1.7782794100389227e-15
gamma: 2.0
spec rad: 1.0
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.0061705331438616385
ld: 1.7782794100389227e-15
gamma: 2.4
spec rad: 1.0
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.0052341796222799015
ld: 3.1622776601683794e-15
gamma: 3.4000000000000004
spec rad: 1.0
s: 0
----------------------------------------
>>> t2 = time()
>>> print(t2-t1)
30358.974792718887

"""
