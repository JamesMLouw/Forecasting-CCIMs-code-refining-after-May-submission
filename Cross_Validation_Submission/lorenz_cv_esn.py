#%%
"""
May 8 Wed: I noticed that h = 0.005 here but in the code for graphing it is h=0.02. I want to change the graphing code and make sure the two correlate (same RC when started with same init cond etc.)
Then run the cross validation code and use the parameters in the graphing code to see if that works. Also check my crossvalidation code.

I also changed the initial condition to (1,1,1), which just looks less picky
"""
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
Z0 = np.array([1,1,1])

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
N = 1000
d_in, d_out = 3, 3

# Construct training input and teacher, testing input and teacher
train_in_esn = data[:ntrain] 
train_teach_esn = data[:ntrain]

# Define the range of parameters for which you want to cross validate over
ld_range = np.logspace(-17,-9,17) 
gamma_range =  np.linspace(2,6,41)
spec_rad_range =  np.linspace(0, 1.5, 16)
s_range = np.array([0])
param_ranges = [ld_range, gamma_range, spec_rad_range, s_range]

# Define the names of the parameters -- orders must match
param_names = ["ld", "gamma", "spec rad", "s"]

# Define the additional inputs taken in by the 
param_add = [N, d_in, d_out, washout]

#%% testing one esn

print(train_teach_esn[11000])

ld = 1e-15
gamma = 3
spec_rad = 1.2
s = 0

esn = ESN(ld, gamma, spec_rad, s, N, d_in, d_out, washout)

print(np.max(esn.A))
print(np.mean(esn.A))
print(np.max(esn.C))
print(np.mean(esn.C))
print(esn.zeta)

#%% train esn

esn.Train(train_in_esn[:12000], use_teacher=False)

w = esn.W
bias = esn.bias
last_state = esn.x_start_path_continue
#%%

print(np.mean(w))
print(np.max(w))
print(w.shape)
print(np.mean(bias))
print(np.max(bias))
print(bias.shape)

print(last_state.shape)
print(np.max(last_state))
print(np.mean(last_state))
#%%

a = np.array([0,1])
a==None
#%%
a = np.array([1,1,1])
b = np.array([1,2,1])
all(a == b)


#%% path continue with esn

predictions = esn.PathContinue(train_in_esn[12000-1], 8000)
last_x = predictions[-1]
print(predictions.shape)
print(np.mean(last_x))
print(np.max(last_x))
print(last_x)


#%%
train_size = 10000
validation_size = 2000
nstarts = 5

if __name__ == "__main__":
    
    t1 = time()
    CV = CrossValidate(validation_parameters=[train_size, validation_size, nstarts],
                       validation_type="rolling", task="PathContinue", norm_type=None)
    print('created CV')
    best_parameters, parameter_combinations, errors = CV.crossvalidate_multiprocessing(ESN,
                                        train_in_esn, train_teach_esn, param_ranges, 
                                        param_names, param_add, num_processes=25, use_target=False)
    t2 = time()
print(t2-t1)

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

#%%
"""
Run on 30 May 2024, Th (not using dictionary, so still un matched with the graphing code)

Best: 
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.003127928205534715
ld: 3.1622776601683796e-14
gamma: 5.4
spec rad: 0.1
s: 0

Grid:


# Define the range of parameters for which you want to cross validate over
ld_range = np.logspace(-17,-9,17) 
gamma_range =  np.linspace(2,6,41)
spec_rad_range =  np.linspace(0, 1.5, 16)
s_range = np.array([0])
param_ranges = [ld_range, gamma_range, spec_rad_range, s_range]


Intermediary Best Parameters:
Validation errors: 793445728.1214458
ld: 1e-17
gamma: 2.0
spec rad: 0.0
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 7628.30702598965
ld: 1e-17
gamma: 2.0
spec rad: 0.1
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 3541.8498236311257
ld: 1e-17
gamma: 2.0
spec rad: 0.2
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 1081.2377806898567
ld: 1e-17
gamma: 2.0
spec rad: 0.5
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 741.758553111379
ld: 1e-17
gamma: 2.0
spec rad: 0.7000000000000001
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 317.5551346114671
ld: 1e-17
gamma: 2.0
spec rad: 1.1
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 92.66233260705368
ld: 1e-17
gamma: 2.0
spec rad: 1.2000000000000002
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.3427711569627728
ld: 1e-17
gamma: 2.0
spec rad: 1.3
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.24292759321201216
ld: 1e-17
gamma: 2.0
spec rad: 1.4000000000000001
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.13836083145541828
ld: 1e-17
gamma: 2.0
spec rad: 1.5
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.13015068423539675
ld: 1e-17
gamma: 2.1
spec rad: 1.4000000000000001
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.12507363973599298
ld: 1e-17
gamma: 2.1
spec rad: 1.5
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.11669376949743475
ld: 1e-17
gamma: 2.2
spec rad: 1.4000000000000001
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.08407107658215418
ld: 1e-17
gamma: 2.3
spec rad: 1.4000000000000001
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.06804031801727349
ld: 1e-17
gamma: 2.5
spec rad: 1.4000000000000001
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.03285399314442734
ld: 1e-17
gamma: 2.9
spec rad: 1.4000000000000001
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.03254627983324654
ld: 1e-17
gamma: 3.1
spec rad: 1.4000000000000001
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.032330541472602906
ld: 1e-17
gamma: 5.4
spec rad: 1.5
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.026219726150601197
ld: 1e-17
gamma: 5.5
spec rad: 1.5
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.025894400337706548
ld: 3.1622776601683796e-17
gamma: 2.2
spec rad: 1.3
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.022759878589686137
ld: 3.1622776601683796e-17
gamma: 2.3
spec rad: 1.3
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.0193883380118021
ld: 3.1622776601683794e-15
gamma: 2.2
spec rad: 0.1
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.01599079347220121
ld: 3.1622776601683794e-15
gamma: 2.2
spec rad: 0.2
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.014755728729427426
ld: 3.1622776601683794e-15
gamma: 2.2
spec rad: 0.30000000000000004
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.01156987464731996
ld: 3.1622776601683794e-15
gamma: 2.3
spec rad: 0.4
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.010688020072204162
ld: 3.1622776601683794e-15
gamma: 2.4
spec rad: 0.5
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.010183410928338504
ld: 3.1622776601683794e-15
gamma: 2.9
spec rad: 0.4
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.009154631179676076
ld: 3.1622776601683794e-15
gamma: 3.0
spec rad: 0.4
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.007635138192081378
ld: 3.1622776601683794e-15
gamma: 3.5
spec rad: 0.2
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.006698542638303479
ld: 3.1622776601683794e-15
gamma: 3.9000000000000004
spec rad: 0.2
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.006600862622208376
ld: 3.1622776601683794e-15
gamma: 4.0
spec rad: 0.1
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.0060040733659155535
ld: 1e-14
gamma: 3.4000000000000004
spec rad: 0.30000000000000004
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.005387244616301604
ld: 1e-14
gamma: 4.300000000000001
spec rad: 0.2
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.00506062979459136
ld: 1e-14
gamma: 4.6
spec rad: 0.30000000000000004
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.0045381968908985975
ld: 1e-14
gamma: 5.2
spec rad: 0.1
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.0036349175215868133
ld: 1e-14
gamma: 5.300000000000001
spec rad: 0.2
s: 0
----------------------------------------
Intermediary Best Parameters:
Validation errors: 0.003127928205534715
ld: 3.1622776601683796e-14
gamma: 5.4
spec rad: 0.1
s: 0
"""