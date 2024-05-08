#%%
import numpy as np
import matplotlib.pyplot as plt
import statistics as stats
from mpl_toolkits.mplot3d import Axes3D 
import numpy.random as random

import sys
# sys.path.append("/Users/JamesLouw/Documents/Personal-Projects-main/[Final] Lorenz")

import data_generate as data_gen
import ESN_helperfunctions as esn_help
import lyapunov_spectrum as lyaps_spec

plt.rcParams['font.size'] = 25

#%% Set parameters for run

access = '$x$ coordinate' # options 'full system', '$x$ coordinate'
#access = 'full system'

experimenting = True #for the names of files to be saved, to prevent overwriting good data

N_init_cond = 10
type_init_conds = 'close' # options 'along trajectory', 'close' or 'far'
spread = 20 # only used in the case of 'along trajectory' to determine how far all the initial conditions should be spread. The initial conditions will be spread evenly from time t=0 up to time t=spread


training_type = 'train over one trajectory'
#training_type = 'train over all trajectories' # options: 'train over one trajectory' or 'train over all trajectories'
# training_type = 'train over all trajectories'


# %% Prepare dataset and initial conditions
def lorenz(t, Z, args):
    u, v, w = Z
    sig, beta, rho = args
    
    up = -sig*(u - v)
    vp = rho*u - v - u*w
    wp = -beta*w + u*v
    
    return np.array([up, vp, wp])

lor_args = (10, 8/3, 28)


if access == 'full system':
    # for access to full system
    h = 0.02
    t_span = (0, 250)
    t_split = 200
    t_trans = 20
elif access == '$x$ coordinate':
    # for access to single coordinate
    h = 0.005
    t_span = (0, 80)
    t_split = 60
    t_trans = 5
else:
     raise ValueError('Choose valid access')


steps_trans = int(t_trans/h)
steps_split = int(t_split/h)

Z0 = (0, 1, 1.05)
N_init_cond = 10
type_init_conds = 'close' # options 'along trajectory', 'close' or 'far'
spread = 20 # only used in the case of 'along trajectory' to determine how far all the initial conditions should be spread. The initial conditions will be spread evenly from time t=0 up to time t=spread

if type_init_conds == 'along trajectory':
    sd = 0
    init_conds = data_gen.gen_init_cond(1, Z0, sd, lorenz, 0, h,
                                    lor_args, pdf = 'gaussian')
else:
    if type_init_conds == 'close':
        sd = 1e-10
    elif type_init_conds == 'far':
        sd = 5
    else:
        raise ValueError('Choose valid type_init_conds')

    init_conds = data_gen.gen_init_cond(N_init_cond, Z0, sd, lorenz, 0, h,
                                        lor_args, pdf = 'gaussian')
    
# save initial conditions
np.save('Initial conditions Lorenz access - ' + access + ' experimenting - ' + str(experimenting) + ' trajectories - ' + str(N_init_cond) + ' type init conds  - ' + type_init_conds + ' sd - ' + str(sd) + ' ',init_conds)

#%% generate trajectories
training_datas = np.zeros( (N_init_cond, steps_split, len(Z0)) ) # to be used for washout and training
testing_datas = np.zeros( (N_init_cond, int(t_span[1]/h) - steps_split, len(Z0)) )

if type_init_conds == 'along trajectory':
    start, end = t_span[0], t_span[1]+spread
    N_data_points = int(t_span[1]/h)
    t_span_temp = (start, end)
    steps_spread = int(spread/N_init_cond/h)
    time_steps, lorenz_datas = data_gen.rk45(lorenz, t_span_temp, init_conds[0], h, lor_args)
    
    for i in range(N_init_cond): 
        training_datas[i] = 0.01*lorenz_datas[i * steps_spread :steps_split+i * steps_spread] 
        testing_datas[i] = 0.01*lorenz_datas[steps_split+i * steps_spread:N_data_points+i * steps_spread]
    time_steps = time_steps[:N_data_points]
    lorenz_datas = lorenz_datas[:N_data_points]
    
elif type_init_conds == 'close' or type_init_conds == 'far':
    for i in range(N_init_cond):
        time_steps, lorenz_datas = data_gen.rk45(lorenz, t_span, init_conds[i], h, lor_args)
        training_datas[i] = 0.01*lorenz_datas[:steps_split] 
        testing_datas[i] = 0.01*lorenz_datas[steps_split:int(t_span[1]/h)]

else:
     raise ValueError('Choose valid type_init_conds')

#save trajectory data

np.save('Training data Lorenz access - full system experimenting - ' + str(experimenting) + ' trajectories - ' + str(N_init_cond) + ' type init conds  - ' + type_init_conds + ' sd - ' + str(sd) + ' ',training_datas)
np.save('Testing data Lorenz access - full system experimenting - ' + str(experimenting) + ' trajectories - ' + str(N_init_cond) + ' type init conds  - ' + type_init_conds + ' sd - ' + str(sd) + ' ',testing_datas)
np.save('Time steps data Lorenz access - ' + access + ' experimenting - ' + str(experimenting) + ' trajectories - ' + str(N_init_cond) + ' type init conds  - ' + type_init_conds + ' sd - ' + str(sd) + ' ',time_steps)

#%% construct observation data

if access == '$x$ coordinate':
    def f(x):
        return x[0] #1/56 * (np.sqrt(1201) - 9) * x[0] + x[1]

    training_datas2 = np.zeros( (N_init_cond, training_datas.shape[1]))
    testing_datas2 = np.zeros( (N_init_cond, testing_datas.shape[1]) )

    for i in range(N_init_cond):        
        training_datas2[i] = np.array([ f(x) for x in training_datas[i] ])
        testing_datas2[i] = np.array([ f(x) for x in testing_datas[i] ])

    del training_datas
    del testing_datas

    training_datas, testing_datas = training_datas2, testing_datas2

    np.save('Training data Lorenz access - ' + str(access) + ' experimenting - ' + str(experimenting) + ' trajectories - ' + str(N_init_cond) + ' type init conds  - ' + type_init_conds + ' sd - ' + str(sd) + ' ',training_datas)
    np.save('Testing data Lorenz access - ' + str(access) + ' experimenting - ' + str(experimenting) + ' trajectories - ' + str(N_init_cond) + ' type init conds  - ' + type_init_conds + ' sd - ' + str(sd) + ' ',testing_datas)

#%% define and train esn

if access == '$x$ coordinate':
	d=1
elif access == 'full system':
	d=3
else:
	raise ValueError('Choose valid access')
	
N = 1000

if access == 'full system':
    # for access to $x$ coord from cross validation We 5 Mar 2024 confirmed with finer grid We 3 Apr 2024

    ld = 3.1622776601683794e-15
    gamma = 13/3
    spec_rad = 1.2
    s = 0

elif access == '$x$ coordinate':
    # for access to $x$ coord from cross validation on Mon 22 Apr 2024

    ld = 3.1622776601683794e-15
    gamma = 3.4
    spec_rad = 1.0
    s = 0

    # for access to $x$ coord second best from cross validation on Mon 22 Apr 2024

    ld = 1.7782794100389227e-15
    gamma = 2.4
    spec_rad = 1.0
    s = 0

else:
    raise ValueError('Choose valid access')

np.random.seed(0) #the random seed 201 works well for the above ld gamma etc.
C = esn_help.gen_matrix(shape=(N,d), density=1, sd=2, mean=-1, pdf="uniform", seeded=False)
A = esn_help.gen_matrix(shape=(N,N), density=0.01, sd=2, mean=-1, pdf="uniform", seeded=False)
A = esn_help.spectral_radius_matrix(A, spec_rad)
zeta = esn_help.gen_matrix(shape=(N,1), density=1, pdf="ones", seeded=False)

x_0 = np.zeros(shape=(N,1), dtype=float)


#%% test for esp
check_esp = False

if check_esp:
    i = random.randint(N_init_cond)
    av_errs = esn_help.test_for_esp(training_datas[i], x_0, 5, A, gamma, C, s, zeta, d, N, 1)

#%% check plot for esp
if check_esp:
    # washout should have about 3000 iterations in it to be safe.
    print(i)
    plt.plot(av_errs[steps_trans:]) #should have very little error and variation here
    plt.show()
   
#%% listen for all trajectories
state_dicts = esn_help.multi_listening(training_datas, x_0, A, gamma, C, s, zeta, d, N)

# save state_dicts
np.save('State dicts Lorenz access - ' + access + ' experimenting - ' + str(experimenting) + ' trajectories - ' + str(N_init_cond) + ' type init conds  - ' + type_init_conds + ' sd - ' + str(sd) + ' ',state_dicts)

#%% train esn

if training_type == 'train over all trajectories':
    reg_result = esn_help.multi_regression_covariance(ld, state_dicts, steps_trans)
elif training_type == 'train over one trajectory':
	reg_result = esn_help.regression_covariance(ld, state_dicts[0], steps_trans)
else:
     raise ValueError('Please pick a training type')

# save regression result

w, bias = reg_result
np.save('Regression result w Lorenz access - ' + access + ' experimenting - ' + str(experimenting) + ' trajectories - ' + str(N_init_cond) + ' type init conds  - ' + type_init_conds + ' sd - ' + str(sd) + ' training type - ' + training_type +' ',w)
np.save('Regression result bias Lorenz access - ' + access + ' experimenting - ' + str(experimenting) + ' trajectories - ' + str(N_init_cond) + ' type init conds  - ' + type_init_conds + ' sd - ' + str(sd) + ' training type - ' + training_type +' ',bias)


#%% predict for all trajectories
	
predictions = esn_help.multi_prediction(state_dicts, reg_result, testing_datas, A, gamma, C, s, zeta, d, N)

#save predictions


np.save('Predictions Lorenz access - ' + access + ' experimenting - ' + str(experimenting) + ' trajectories - ' + str(N_init_cond) + ' type init conds  - ' + type_init_conds + ' sd - ' + str(sd) + ' training type - ' + training_type +' ',predictions)

av_training_error = esn_help.av_multi_training_error(state_dicts, reg_result, steps_trans)
print('Average training error: ', av_training_error)

#save average training error

np.save('Average training error Lorenz access - ' + access + ' experimenting - ' + str(experimenting) + ' trajectories - ' + str(N_init_cond) + ' type init conds  - ' + type_init_conds + ' sd - ' + str(sd) + ' training type - ' + training_type +' ',av_training_error)

error = 0
for i in range(N_init_cond):
    error += predictions[i]['testing_error']
av_error = error/N_init_cond

print('Average testing error: ', av_error)

# save average testing error

np.save('Average testing error Lorenz access - ' + access + ' experimenting - ' + str(experimenting) + ' trajectories - ' + str(N_init_cond) + ' type init conds  - ' + type_init_conds + ' sd - ' + str(sd) + ' training type - ' + training_type +' ',av_error)

# %% plot lyapunov spectrums
skip_les = False

if not(skip_les):
    lyapunov_trajectory = 'same' #'same' or 'long'

    if lyapunov_trajectory == 'same':
        # use same trajectory as for training of the esn
        x_data = predictions[0]['states'].transpose()

    elif lyapunov_trajectory == 'long':
        # create a longer trajectory for calculating LEs
        t_span_le = (0,300)
        time_steps_le, lorenz_data_le = data_gen.rk45(lorenz, t_span_le, init_conds[0], h, lor_args)
        training_data_le, testing_data_le = 0.01*lorenz_data_le[:steps_split], 0.01*lorenz_data_le[steps_split:]
        
        if access == '$x$ coordinate':
        
            training_data_le2 = np.zeros( (N_init_cond, len(training_data_le[0])) )
            testing_data_le2 = np.zeros( (N_init_cond, len(testing_data_le[0])) )

            training_data_le2 = np.array([ f(x) for x in training_data_le ])
            testing_data_le2 = np.array([ f(x) for x in testing_data_le ])

            del training_data_le
            del testing_data_le

            training_data_le, testing_data_le = training_data_le2, testing_data_le2   

        state_dict_le = esn_help.listening(training_data_le, x_0, A, gamma, C, s, zeta, d, N)
        predicting_le = esn_help.prediction(state_dict_le, reg_result, testing_data_le, A, gamma, C, s, zeta, d, N)

        x_data = predicting_le['states'].transpose()

    else:
        raise ValueError('Choose trajectory type for calulcating LEs')

    weights = reg_result[0]
    N_transient = 0

    def jacobian_esn_func(t, x_t_1):
        outer = (1 - np.power(x_t_1, 2)).reshape(N, 1)
        J0 = weights.T  
        J1 = outer * A 
        J2 = outer * C * gamma 
        return J1 + J2 @ J0

#%% run algorithm to calculate les
only_top = False
if not(skip_les):
    if only_top:
        %time lorenz_esn_top_le, lorenz_esn_time = lyaps_spec.top_lyap_exp(x_data, N, h, "difference", 0, jacobian_esn_func, seed = 0,  normalise = 10, t_cap = 50000)
    else:
        %time lorenz_esn_spectrum, lorenz_esn_time = lyaps_spec.lyapunov_spectrum(x_data, N, h, "difference", 0, jacobian_esn_func, seed = 0, gram_schmidt = 10, t_cap = 50000, lyap_store = 1)
        lorenz_esn_top_le = max(lorenz_esn_spectrum)

    # save top le

    np.save('Top LE ESN Lorenz access - ' + access + ' experimenting - ' + str(experimenting) + ' trajectories - ' + str(N_init_cond) + ' type init conds  - ' + type_init_conds + ' sd - ' + str(sd) + ' training type - ' + training_type +' ',lorenz_esn_top_le)
    np.save('LE convergence Lorenz access - ' + access + ' experimenting - ' + str(experimenting) + ' trajectories - ' + str(N_init_cond) + ' type init conds  - ' + type_init_conds + ' sd - ' + str(sd) + ' training type - ' + training_type +' ',lorenz_esn_time)
    
else:
     lorenz_esn_top_le = 0.9125459202784676 #for parameters on 8 April 2024 Monday
     # np.save('Top LE ESN Lorenz access - ' + access + ' experimenting - ' + str(experimenting) + ' trajectories - ' + str(N_init_cond) + ' type init conds  - ' + type_init_conds + ' sd - ' + str(sd) + ' training type - ' + training_type +' ',lorenz_esn_top_le)

#%%
print(lorenz_esn_top_le)

#%%//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#%% Graphing

#%%//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#%% Read in data

init_conds = np.load('Initial conditions Lorenz access - ' + access + ' experimenting - ' + str(experimenting) + ' trajectories - ' + str(N_init_cond) + ' type init conds  - ' + type_init_conds + ' sd - ' + str(sd) + ' .npy')
training_datas = np.load('Training data Lorenz access - ' + access + ' experimenting - ' + str(experimenting) + ' trajectories - ' + str(N_init_cond) + ' type init conds  - ' + type_init_conds + ' sd - ' + str(sd) + ' .npy')
testing_datas =  np.load('Testing data Lorenz access - ' + access + ' experimenting - ' + str(experimenting) + ' trajectories - ' + str(N_init_cond) + ' type init conds  - ' + type_init_conds + ' sd - ' + str(sd) + ' .npy')
time_steps =  np.load('Time steps data Lorenz access - ' + access + ' experimenting - ' + str(experimenting) + ' trajectories - ' + str(N_init_cond) + ' type init conds  - ' + type_init_conds + ' sd - ' + str(sd) + ' .npy')
state_dicts = np.load('State dicts Lorenz access - ' + access + ' experimenting - ' + str(experimenting) + ' trajectories - ' + str(N_init_cond) + ' type init conds  - ' + type_init_conds + ' sd - ' + str(sd) + ' .npy', allow_pickle = True)

w = np.load('Regression result w Lorenz access - ' + access + ' experimenting - ' + str(experimenting) + ' trajectories - ' + str(N_init_cond) + ' type init conds  - ' + type_init_conds + ' sd - ' + str(sd) + ' training type - ' + training_type + ' .npy')
bias = np.load('Regression result bias Lorenz access - ' + access + ' experimenting - ' + str(experimenting) + ' trajectories - ' + str(N_init_cond) + ' type init conds  - ' + type_init_conds + ' sd - ' + str(sd) + ' training type - ' + training_type + ' .npy')
reg_result = (w, bias)
predictions = np.load('Predictions Lorenz access - ' + access + ' experimenting - ' + str(experimenting) + ' trajectories - ' + str(N_init_cond) + ' type init conds  - ' + type_init_conds + ' sd - ' + str(sd) + ' training type - ' + training_type + ' .npy', allow_pickle = True)
av_training_error = np.load('Average training error Lorenz access - ' + access + ' experimenting - ' + str(experimenting) + ' trajectories - ' + str(N_init_cond) + ' type init conds  - ' + type_init_conds + ' sd - ' + str(sd) + ' training type - ' + training_type + ' .npy')
av_error = np.load('Average testing error Lorenz access - ' + access + ' experimenting - ' + str(experimenting) + ' trajectories - ' + str(N_init_cond) + ' type init conds  - ' + type_init_conds + ' sd - ' + str(sd) + ' training type - ' + training_type + ' .npy')
lorenz_esn_top_le = np.load('Top LE ESN Lorenz access - ' + access + ' experimenting - ' + str(experimenting) + ' trajectories - ' + str(N_init_cond) + ' type init conds  - ' + type_init_conds + ' sd - ' + str(sd) + ' training type - ' + training_type + ' .npy')
lorenz_esn_time = np.load('LE convergence Lorenz access - ' + access + ' experimenting - ' + str(experimenting) + ' trajectories - ' + str(N_init_cond) + ' type init conds  - ' + type_init_conds + ' sd - ' + str(sd) + ' training type - ' + training_type +' .npy')

#%%
print(lorenz_esn_top_le)
#%%
plt.plot(lorenz_esn_time)
#%%
error_bound = [t*l for (t,l) in zip(range(1000), lorenz_esn_time[:1000])]
plt.plot(error_bound)

#%%

diff = np.abs( predicting['z_actuals'][0] - predicting['z_predictions'][0]) 
plt.plot(np.log(diff[:1000]), lw = 0.7)
#%%
lorenz_esn_top_le = lorenz_esn_time[100]
# %% check a forecast and its errors
i = random.randint(N_init_cond)
i=0
print(i)

if access == '$x$ coordinate':
    predicting = predictions[i]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 15), tight_layout=True)
    #fig.suptitle("Lorenz predictions against actual")

    ax1.plot(predicting['z_actuals'][0], label = 'True trajectory')
    ax1.plot(predicting['z_predictions'][0], label = 'Forecasted trajectory')
    ax1.set_ylabel('x')
    ax1.set_title('(a)')
    ax1.set_xlim(0,4000)
    #ax1.legend()
    
    diff = np.abs( predicting['z_actuals'][0] - predicting['z_predictions'][0]) 

    ax2.plot(diff)
    ax2.set_ylabel('errors')
    ax2.set_title('(b)')
    ax3.set_xlim(0,4000)

    ax3.plot(np.log(diff))
    ax3.set_ylabel(' log errors ')
    ax3.set_title('(c)')
    ax3.set_xlim(0,4000)

    fig.show()

elif access == 'full system':
    i = random.randint(0,N_init_cond)
    print(i)
    u = 1 * (predictions[i]['z_predictions'][0] - testing_datas[i].transpose()[0])
    v = 1 * (predictions[i]['z_predictions'][1] - testing_datas[i].transpose()[1])
    w = 1 * (predictions[i]['z_predictions'][2] - testing_datas[i].transpose()[2])

    start = 0
    stop = testing_datas.shape[1]
    step = 1

    plot_ls = time_steps[start:stop:step]

    u_plot = u[start:stop:step]
    v_plot = v[start:stop:step]
    w_plot = w[start:stop:step]

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(u_plot, v_plot, w_plot, c=plot_ls, cmap='viridis', s = 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Error data')
    plt.show()


#%% plot average prediction error for one trajectory
print(lorenz_esn_top_le)
i = random.randint(N_init_cond)
print(i)

title = 'system: Lorenz, access: '+ access + ', trajectories: '+ str(1) + ', type init conds: ' + type_init_conds + ', sd: ' + str(sd) + ', training: ' + training_type
z_errors = esn_help.average_error_traj([predictions[i]]) #[random.choice(predictions)])
esn_help.prediction_error_plot(z_errors, h, lorenz_esn_top_le,
plot_with_error_bound = True, log_scale = True, time_cut_off=20, N_transient=180,
title = title, include_title = True)

#%% plot average prediction error

predictionsi = predictions[::1]
z_errors = esn_help.average_error_traj(predictions)
title = 'system: Lorenz, access: '+ access + ', trajectories: '+ str(N_init_cond) + ', type init conds: ' + type_init_conds + ', sd: ' + str(np.log(sd)) + ', training: ' + training_type
title = ''
esn_help.prediction_error_plot(z_errors, h, lorenz_esn_top_le,
plot_with_error_bound = True, log_scale = True, time_cut_off=10, N_transient=400,
title = title, include_title = True)
print('type init conds:', type_init_conds)
print('training type:', training_type)


# %% Check and plot training dataset
if access == 'full system':
    # check training data
    i = random.randint(0,N_init_cond)
    print(i)
    u, v, w = training_datas[i].transpose()

    start = steps_trans
    stop = training_datas.shape[1] 
    step = 1

    plot_ls = time_steps[start:stop:step]

    u_plot = u[start:stop:step]
    v_plot = v[start:stop:step]
    w_plot = w[start:stop:step]

    #fig, ax = plt.subplots(1,N_init_cond+1)
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.scatter(u_plot, v_plot, w_plot, c=plot_ls, cmap='viridis', s = 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #ax.set_title('The Lorenz attractor')

# %% Check and plot a few training datasets
if access == 'full system':
    start = steps_trans
    stop = training_datas.shape[1] 
    step = 1

    plot_ls = time_steps[start:stop:step]
    fig = plt.figure(figsize=(15, 10))

    for j in range(min(4,N_init_cond)):
        u, v, w = training_datas[j].transpose()
        u_plot = u[start:stop:step]
        v_plot = v[start:stop:step]
        w_plot = w[start:stop:step]
        
        ax = fig.add_subplot(2,2,j+1, projection='3d')
        ax.scatter(u_plot, v_plot, w_plot, c=plot_ls, cmap='viridis', s = 5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('Training data '+str(j))

    plt.show()

# %% check testing data
if access == 'full system':
    i = random.randint(0,N_init_cond)
    print(i)
    u, v, w = testing_datas[i].transpose()

    start = 0
    stop = testing_datas.shape[1] 
    step = 1

    plot_ls = time_steps[start:stop:step]

    u_plot = u[start:stop:step]
    v_plot = v[start:stop:step]
    w_plot = w[start:stop:step]

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(u_plot, v_plot, w_plot, c=plot_ls, cmap='viridis', s = 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Testing data')
    plt.show()

# %% Check and plot a few testing datasets
if access == 'full system':
    start = steps_trans
    stop = testing_datas.shape[1] 
    step = 1

    plot_ls = time_steps[start:stop:step]
    fig = plt.figure(figsize=(15, 10))

    for j in range(min(4,N_init_cond)):
        u, v, w = testing_datas[j].transpose()
        u_plot = u[start:stop:step]
        v_plot = v[start:stop:step]
        w_plot = w[start:stop:step]
        
        ax = fig.add_subplot(2,2,j+1, projection='3d')
        ax.scatter(u_plot, v_plot, w_plot, c=plot_ls, cmap='viridis', s = 5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('Training data '+str(j))

    plt.show()

# %% check time series
if access == 'full system':
    i = np.random.randint(N_init_cond)
    print(i)
    u, v, w = np.concatenate([training_datas[i], testing_datas[i]]).transpose()

    start = 0
    stop = len(u)
    step = 1

    u_plot = u[start:stop:step]
    v_plot = v[start:stop:step]
    w_plot = w[start:stop:step]

    plot_ls = time_steps[start:stop:step]


    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 15), tight_layout=True)
    ax1.axvline(t_split, color='black', linestyle="solid")
    ax1.scatter(time_steps[start:stop:step], u_plot, c=plot_ls, cmap='viridis', s=2)
    ax1.set_ylabel('x')
    ax2.scatter(time_steps[start:stop:step], v_plot, c=plot_ls, cmap='viridis', s=2)
    ax2.axvline(t_split, color='black', linestyle="solid")
    ax2.set_ylabel('y')
    ax3.scatter(time_steps[start:stop:step], w_plot, c=plot_ls, cmap='viridis', s=2)
    ax3.axvline(t_split, color='black', linestyle="solid")
    ax3.set_ylabel('z')
    ax3.set_xlabel('time')

    plt.show()


# %% check time series of transformed data for training

if access == '$x$ coordinate':
    i = np.random.randint(N_init_cond)
    print(i)
    u = np.concatenate((training_datas[i], testing_datas[i]))

    start = 0
    stop = len(u)
    step = 1

    u_plot = u[start:stop:step]
    plot_ls = time_steps[start:stop:step]

    print(u_plot.shape, plot_ls.shape)

    fig, ax1 = plt.subplots(1, 1, figsize=(20, 7), tight_layout=True)
    ax1.axvline(t_split, color='black', linestyle="--")
    ax1.scatter(plot_ls, u_plot, c=plot_ls, cmap='viridis', s=3)
    ax1.set_ylabel('x')

    plt.show()
    
# %% check historical accuracy plots

predicting = predictions[0]
hist_u_actuals = predicting['z_actuals'][0]
hist_u_predictions = predicting['z_predictions'][0]
esn_help.hist_accuracy_plot(hist_u_actuals, hist_u_predictions, x_label='x', y_label='frequency')

# %% check historical accuracy plots for another trajectory

i = random.randint(0,N_init_cond)
print(i)
hist_u_actuals = predictions[i]['z_actuals'][0]
hist_u_predictions = predictions[i]['z_predictions'][0]
esn_help.hist_accuracy_plot(hist_u_actuals, hist_u_predictions, x_label='x', y_label='frequency')

#%% check historical accuracy plots
if access == 'full system':
    hist_v_actuals = predicting['z_actuals'][1]
    hist_v_predictions = predicting['z_predictions'][1]
    esn_help.hist_accuracy_plot(hist_v_actuals, hist_v_predictions, x_label='y', y_label='frequency')

# %% check historical accuracy plots for another trajectory

if access == 'full system':
    print(i)
    hist_u_actuals = predictions[i]['z_actuals'][1]
    hist_u_predictions = predictions[i]['z_predictions'][1]
    esn_help.hist_accuracy_plot(hist_u_actuals, hist_u_predictions, x_label='y', y_label='frequency')

# %% check historical accuracy plots

if access == 'full system':
    predicting = predictions[0]
    hist_w_actuals = predicting['z_actuals'][2]
    hist_w_predictions = predicting['z_predictions'][2]
    esn_help.hist_accuracy_plot(hist_w_actuals, hist_w_predictions, x_label='z', y_label='frequency')

# %% check historical accuracy plots for another trajectory

if access == 'full system':
    print(i)
    hist_u_actuals = predictions[i]['z_actuals'][2]
    hist_u_predictions = predictions[i]['z_predictions'][2]
    esn_help.hist_accuracy_plot(hist_u_actuals, hist_u_predictions, x_label='z', y_label='frequency')

# %% historical accuracy plots -- all 3 dimensions

if access == 'full system':
    hist_actuals = np.concatenate((predicting['z_actuals'][0], predicting['z_actuals'][1], predicting['z_actuals'][2]), axis=0)
    hist_predictions = np.concatenate((predicting['z_predictions'][0], predicting['z_predictions'][1], predicting['z_predictions'][2]), axis=0)
    esn_help.hist_accuracy_plot(hist_actuals, hist_predictions, x_label='all', y_label='frequency')

# %% historical accuracy plots -- all 3 dimensions

if access == 'full system':
    print(i)
    hist_actuals = np.concatenate((predictions[i]['z_actuals'][0], predictions[i]['z_actuals'][1], predictions[i]['z_actuals'][2]), axis=0)
    hist_predictions = np.concatenate((predictions[i]['z_predictions'][0], predictions[i]['z_predictions'][1], predictions[i]['z_predictions'][2]), axis=0)
    esn_help.hist_accuracy_plot(hist_actuals, hist_predictions, x_label='all', y_label='frequency')

# %% check attractor reconstruction

i = random.randint(0,N_init_cond)
print(i)

if access == 'full system':
    u = predictions[i]['z_predictions'][0]
    v = predictions[i]['z_predictions'][1]
    w = predictions[i]['z_predictions'][2]

    start = 0
    stop = testing_datas.shape
    step = 1

elif access == '$x$ coordinate':
    lag = 10
    u = predictions[i]['z_predictions'][0][:-2*lag]
    v = predictions[i]['z_predictions'][0][lag:-lag]
    w = predictions[i]['z_predictions'][0][2*lag:]

    start = 0
    stop = testing_datas.shape[1] - 2*lag
    step = 1


plot_ls = time_steps[start:stop:step]

u_plot = u[start:stop:step]
v_plot = v[start:stop:step]
w_plot = w[start:stop:step]

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(u_plot, v_plot, w_plot, c=plot_ls, cmap='viridis', s = 5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Predicted data')
plt.show()


# %% generate actual Lyapunov spectrum for lorenz system
skip_system_les = True

if not(skip_system_les):   
    steps_trans_lor = int(20/h)

    def jacobian_lorenz(t, lor_inputs):
        u, v, w = lor_inputs
        sig, beta, rho = 10, 8/3, 28
        
        dF1dinputs = [-sig, sig, 0]
        dF2dinputs = [rho-w, -1, -u]
        dF3dinputs = [v, u, -beta]
        
        return np.array([dF1dinputs, dF2dinputs, dF3dinputs])

only_top_system = True
if not(skip_system_les):
    if only_top_system:
        %time lorenz_top_le = lyaps_spec.top_lyap_exp(lorenz_data, d, h, "ordinary", 2*steps_trans_lor, jacobian_lorenz, t_cap = 50000)
    else:
        %time lorenz_spectrum = lyaps_spec.lyapunov_spectrum(lorenz_data, d, h, "ordinary", steps_trans_lor, jacobian_lorenz, gram_schmidt = 30, t_cap = 50000, lyap_store = 1))
        lorenz_top_le = max(lorenz_spectrum)

else:
     lorenz_top_le = 0.921353628

print(lorenz_top_le)

# %% plot spectrums
if not(only_top) and not(only_top_system):
    plt.figure(figsize=(10, 5))

    lor_spectrum_sorted = np.sort(lorenz_spectrum)[::-1]
    esn_spectrum_sorted = np.sort(lorenz_esn_spectrum)[::-1]
    esn_spectrum_sorted = esn_spectrum_sorted[:100]

    mg_idx = np.arange(0, len(lor_spectrum_sorted))
    plot_mg = lor_spectrum_sorted
    plt.scatter(mg_idx, plot_mg, s=10, marker='o', label='actual')

    esn_idx = np.arange(0, len(esn_spectrum_sorted))
    plot_mg_esn = esn_spectrum_sorted
    plt.scatter(esn_idx, plot_mg_esn, s=0.7, marker='x', label='ESN')

    plt.axhline(c='black', lw=1, linestyle='--')

    plt.ylabel('$\lambda$')
    plt.xlabel('dimension')
    plt.legend()

# %% Kaplan-Yorke dimension
if not(only_top) and not(only_top_system):
    KY_Lor = lyaps_spec.kaplan_yorke(lor_spectrum_sorted)
    KY_esn = lyaps_spec.kaplan_yorke(esn_spectrum_sorted)

    print(KY_Lor, KY_esn)


#%%//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#%% Graphing for attractors from full system

#%%//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#%% Read in data
access = 'full system'
training_datas = np.load('Training data Lorenz access - ' + access + ' experimenting - ' + str(experimenting) + ' trajectories - ' + str(N_init_cond) + ' type init conds  - ' + type_init_conds + ' sd - ' + str(sd) + ' .npy')
testing_datas =  np.load('Testing data Lorenz access - ' + access + ' experimenting - ' + str(experimenting) + ' trajectories - ' + str(N_init_cond) + ' type init conds  - ' + type_init_conds + ' sd - ' + str(sd) + ' .npy')

# %% Check and plot training dataset
if access == 'full system':
    # check training data
    i = random.randint(0,N_init_cond)
    print(i)
    u, v, w = training_datas[i].transpose()

    start = steps_trans
    stop = training_datas.shape[1] 
    step = 1

    plot_ls = time_steps[start:stop:step]

    u_plot = u[start:stop:step]
    v_plot = v[start:stop:step]
    w_plot = w[start:stop:step]

    #fig, ax = plt.subplots(1,N_init_cond+1)
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.scatter(u_plot, v_plot, w_plot, c=plot_ls, cmap='viridis', s = 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #ax.set_title('The Lorenz attractor')

# %% Check and plot a few training datasets
if access == 'full system':
    start = steps_trans
    stop = training_datas.shape[1] 
    step = 1

    plot_ls = time_steps[start:stop:step]
    fig = plt.figure(figsize=(15, 10))

    for j in range(min(4,N_init_cond)):
        u, v, w = training_datas[j].transpose()
        u_plot = u[start:stop:step]
        v_plot = v[start:stop:step]
        w_plot = w[start:stop:step]
        
        ax = fig.add_subplot(2,2,j+1, projection='3d')
        ax.scatter(u_plot, v_plot, w_plot, c=plot_ls, cmap='viridis', s = 5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('Training data '+str(j))

    plt.show()

# %% check testing data
if access == 'full system':
    i = random.randint(0,N_init_cond)
    print(i)
    u, v, w = testing_datas[i].transpose()

    start = 0
    stop = testing_datas.shape[1] 
    step = 1

    plot_ls = time_steps[start:stop:step]

    u_plot = u[start:stop:step]
    v_plot = v[start:stop:step]
    w_plot = w[start:stop:step]

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(u_plot, v_plot, w_plot, c=plot_ls, cmap='viridis', s = 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Testing data')
    plt.show()

# %% Check and plot a few testing datasets
if access == 'full system':
    start = steps_trans
    stop = testing_datas.shape[1] 
    step = 1

    plot_ls = time_steps[start:stop:step]
    fig = plt.figure(figsize=(15, 10))

    for j in range(min(4,N_init_cond)):
        u, v, w = testing_datas[j].transpose()
        u_plot = u[start:stop:step]
        v_plot = v[start:stop:step]
        w_plot = w[start:stop:step]
        
        ax = fig.add_subplot(2,2,j+1, projection='3d')
        ax.scatter(u_plot, v_plot, w_plot, c=plot_ls, cmap='viridis', s = 5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('Training data '+str(j))

    plt.show()

# %% check time series
if access == 'full system':
    i = np.random.randint(N_init_cond)
    print(i)
    u, v, w = np.concatenate([training_datas[i], testing_datas[i]]).transpose()

    start = 0
    stop = len(u)
    step = 1

    u_plot = u[start:stop:step]
    v_plot = v[start:stop:step]
    w_plot = w[start:stop:step]

    plot_ls = time_steps[start:stop:step]


    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 15), tight_layout=True)
    ax1.axvline(t_split, color='black', linestyle="solid")
    ax1.scatter(time_steps[start:stop:step], u_plot, c=plot_ls, cmap='viridis', s=2)
    ax1.set_ylabel('x')
    ax2.scatter(time_steps[start:stop:step], v_plot, c=plot_ls, cmap='viridis', s=2)
    ax2.axvline(t_split, color='black', linestyle="solid")
    ax2.set_ylabel('y')
    ax3.scatter(time_steps[start:stop:step], w_plot, c=plot_ls, cmap='viridis', s=2)
    ax3.axvline(t_split, color='black', linestyle="solid")
    ax3.set_ylabel('z')
    ax3.set_xlabel('time')

    plt.show()
# %%
