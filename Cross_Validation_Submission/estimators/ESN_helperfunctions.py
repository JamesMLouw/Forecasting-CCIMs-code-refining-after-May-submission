import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import random
from scipy import stats
from statistics import mean
import seaborn as sns

plt.rcParams['font.size'] = 25


# Function to generate reservoir

def gen_matrix(shape, density, sd=1, mean=0, loc_seed=100, val_seed=100, pdf="gaussian", seeded=True):
    
    def seeded_rvs_gauss(array_len):
            return stats.norm(loc=mean, scale=sd).rvs(random_state = val_seed, size=array_len)

    def seeded_rvs_uniform(array_len):
        return stats.uniform(loc=mean, scale=sd).rvs(random_state = val_seed, size=array_len)

    m = shape[0]
    n = shape[1]

    if seeded == True:
        
        if pdf == "gaussian":
            M = random(m, n, density=density, random_state=loc_seed, data_rvs=seeded_rvs_gauss).A
            return M

        if pdf == "uniform":
            M = random(m, n, density=density, random_state=loc_seed, data_rvs=seeded_rvs_uniform).A
            return M

        if pdf == "ones":
            M = random(m, n, density=density, random_state=loc_seed, data_rvs=np.ones).A
            return M
        else: 
            print("No such pdf")
            
    elif seeded == False:
        
        if pdf == "gaussian":
            unseeded_rvs = stats.norm(loc=mean, scale=sd).rvs
            M = random(m, n, density=density, data_rvs=unseeded_rvs).A
            return M

        if pdf == "uniform":
            unseeded_rvs = stats.uniform(loc=mean, scale=sd).rvs
            M = random(m, n, density=density, data_rvs=unseeded_rvs).A
            return M

        if pdf == "ones":
            M = random(m, n, density=density, data_rvs=np.ones).A
            return M
        else: 
            print("No such pdf")
            
    else:
        print("Seeded was neither true nor false")

def spectral_radius(M):
    max_abs_eigenvalue = -1
    eigenvalues, eigenvectors = np.linalg.eig(M)
    for eigenvalue in eigenvalues:
        if abs(eigenvalue) > max_abs_eigenvalue:
            max_abs_eigenvalue = abs(eigenvalue)
    return max_abs_eigenvalue

def spectral_radius_matrix(M, desired_spec_rad):
    M_sr = spectral_radius(M)
    if M_sr == 0:
        return M
    else:
        M = M*(desired_spec_rad/M_sr)
        return M    
    
def norm_matrix(M, desired_norm):
    M_n = np.linalg.norm(M)
    if M_n == 0:
        return M
    else:
        return M * (desired_norm/M_n)
    
# Reservoir equations

def sigma(value):
    return np.tanh(value)

def state(x_prev, z_curr, A, gamma, C, s, zeta, d):
    x_curr = sigma(A @ x_prev + gamma*(C @ z_curr) + s*zeta)
    return x_curr    

def observation(x_curr, reg_result):
    w, bias = reg_result
    z_next = np.transpose(w) @ x_curr + bias
    return z_next    

# Listening, training and predicting(testing)

def listening(training_data, x_0, A, gamma, C, s, zeta, d, N):
    state_dict = {'all_states': None,
                  'last_state': None, 
                  'input_data': None}
    
    T = len(training_data)
    
    X = np.zeros((N, T))
    Z = np.zeros((d, T))
   
    
    for t in range(0, T):
        if t == 0:
            x_curr = x_0
            z_curr = training_data[t].reshape(d, 1)
            
            X[:, 0] = x_curr[:, 0]
            Z[:, 0] = z_curr[:, 0]
            
        else:
            x_curr = state(x_curr, z_curr, A, gamma, C, s, zeta, d)
            z_curr = training_data[t].reshape(d, 1)
            
            X[:, t] = x_curr[:, 0]
            Z[:, t] = z_curr[:, 0]
                
    state_dict['last_state'] = x_curr # this is x_T-1, where our training data runs up to time T-1
    state_dict['all_states'] = X
    state_dict['input_data'] = Z
    
    return state_dict

def multi_listening(training_datas, x_0, A, gamma, C, s, zeta, d, N):
    state_dicts = []
    
    for i in range(len(training_datas)):
        print(i)
        state_dicts = state_dicts + [listening(training_datas[i], x_0, A, gamma, C, s, zeta, d, N)]
        
    return state_dicts
        
def regression_sol_alt(ld, state_dict, T_trans):
    X = state_dict['all_states'][:, T_trans:]
    Z = state_dict['input_data'][:, T_trans:]
   
    N = X.shape[0]
    d = Z.shape[0]
    
    w_best = np.linalg.solve(X @ X.transpose() + ld * np.identity(N), X @ Z.transpose())
    a_best = (np.mean(Z, axis=1) - (w_best.transpose() @ np.mean(X, axis=1).reshape(N, 1))).reshape(d, 1)
    
    return w_best, a_best    # outputs (N, d) array and (1, ) array


def regression_sol(ld, state_dict, T_trans):
    X = state_dict['all_states'][:, T_trans:]
    Z = state_dict['input_data'][:, T_trans:]

    N = X.shape[0]
    T = X.shape[1]
    d = Z.shape[0]
    
    X_concat = np.concatenate((X, np.ones(shape=(1, T))), axis=0)
    X_tranpose_concat = np.concatenate((X.transpose(), np.zeros(shape=(T, 1))), axis=1)
    
    regularisation = ld * np.identity(N)
    zeros_row = np.zeros(shape=(1, N))
    zeros_col = np.zeros(shape=(N+1, 1))
    regularisation_concat = np.concatenate((regularisation, zeros_row), axis=0)
    regularisation_concat = np.concatenate((regularisation_concat, zeros_col), axis=1) 
    regularisation_concat[N][N] = T
    
    reg_best = np.linalg.solve(X_concat @ X_tranpose_concat + regularisation_concat, X_concat @ Z.transpose())
    
    w_best = reg_best[0:N, :]
    a_best = reg_best[N, :].reshape(d, 1)
    
    return w_best, a_best    # outputs same forecast results as regression_sol

def regression_covariance(ld, state_dict, T_trans):
    X = state_dict['all_states'][:, T_trans:]
    Z = state_dict['input_data'][:, T_trans:]
    
    N = X.shape[0]
    T = X.shape[1]
    d = Z.shape[0]
    
    cov_XZ = (1/T) * (X @ Z.transpose()) - (np.mean(X, axis=1).reshape(N, 1) @ np.mean(Z, axis=1).reshape(d, 1).transpose())
    cov_XX = (1/T) * (X @ X.transpose()) - (np.mean(X, axis=1).reshape(N, 1) @ np.mean(X, axis=1).reshape(N, 1).transpose()) + ld * np.identity(N)
    
    w_best = np.linalg.solve(cov_XX, cov_XZ)
    a_best = (np.mean(Z, axis=1) - (w_best.transpose() @ np.mean(X, axis=1))).reshape(d, 1)

    return w_best, a_best

def regression_covariance_targets(ld, states, targets, washout):
    X = states[:, washout:]
    Z = targets[:, washout:]
    N = X.shape[0]
    T = X.shape[1]
    d = Z.shape[0]

    print('X', X.shape)
    print(X[:5,-1])
    print('Z', Z.shape)
    print(Z[:,-1])
    print('lambda', ld)
    print(type(Z))
    
    cov_XZ = (1/T) * (X @ Z.transpose()) - (np.mean(X, axis=1).reshape(N, 1) @ np.mean(Z, axis=1).reshape(d, 1).transpose())
    cov_XX = (1/T) * (X @ X.transpose()) - (np.mean(X, axis=1).reshape(N, 1) @ np.mean(X, axis=1).reshape(N, 1).transpose()) + ld * np.identity(N)

    print(cov_XZ.shape)
    print('cov_XZ', cov_XZ[5,:])
    
    w_best = np.linalg.solve(cov_XX, cov_XZ)
    a_best = (np.mean(Z, axis=1) - (w_best.transpose() @ np.mean(X, axis=1))).reshape(d, 1)

    return w_best, a_best

def multi_regression_covariance(ld, state_dicts, T_trans):
    X = []
    Z = []
    
    for i in range(len(state_dicts)):
        X = X + [state_dicts[i]['all_states'][:, T_trans:].transpose()]
        Z = Z + [state_dicts[i]['input_data'][:, T_trans:].transpose()]
    
    X = np.concatenate( X ).transpose()
    Z = np.concatenate( Z ).transpose()
    
    print('X', X.shape)
    print(X[:5,-1])
    print('Z', Z.shape)
    print(Z[:,-1])
    print('lambda', ld)

    
    N = X.shape[0]
    T = X.shape[1]
    d = Z.shape[0]
    
    cov_XZ = (1/T) * (X @ Z.transpose()) - (np.mean(X, axis=1).reshape(N, 1) @ np.mean(Z, axis=1).reshape(d, 1).transpose())
    cov_XX = (1/T) * (X @ X.transpose()) - (np.mean(X, axis=1).reshape(N, 1) @ np.mean(X, axis=1).reshape(N, 1).transpose()) + ld * np.identity(N)
    
    w_best = np.linalg.solve(cov_XX, cov_XZ)
    a_best = (np.mean(Z, axis=1) - (w_best.transpose() @ np.mean(X, axis=1))).reshape(d, 1)

    return w_best, a_best


def prediction(state_dict, reg_result, testing_data, A, gamma, C, s, zeta, d, N):
    prediction_dict = {'testing_error': None,
                       'z_actuals': None,
                       'z_predictions': None,
                       'states': None}
    
    T_bar = len(testing_data)
    x_prev = state_dict.get('last_state')
    z_predict = state_dict.get('input_data')[:, -1].reshape(d, 1)
    
    testing_error = 0
    z_predictions = np.zeros(shape=(d, T_bar))
    z_actuals = np.zeros(shape=(d, T_bar))
    states = np.zeros(shape=(N, T_bar))
    
    for t_bar in range(0, T_bar):
        x_prev = state(x_prev, z_predict, A, gamma, C, s, zeta, d)
        z_predict = observation(x_prev, reg_result)
        z_actual = np.array(testing_data[t_bar]).reshape(d, 1)
        
        testing_error = testing_error + np.linalg.norm(z_predict - z_actual)**2
        z_predictions[:, t_bar] = z_predict[:, 0]
        z_actuals[:, t_bar] = z_actual[:, 0]
        states[:, t_bar] = x_prev[:, 0]
    
    prediction_dict['testing_error'] = testing_error / T_bar
    prediction_dict['z_actuals'] = z_actuals
    prediction_dict['z_predictions'] = z_predictions   
    prediction_dict['states'] = states
        
    return prediction_dict

def multi_prediction(state_dicts, reg_result, testing_datas, A, gamma, C, s, zeta, d, N):
    prediction_dicts = []
    
    for i in range(len(state_dicts)):
        print(i)
        prediction_dicts = prediction_dicts + [prediction(state_dicts[i], reg_result, testing_datas[i], A, gamma, C, s, zeta, d, N)]
        
    return prediction_dicts

def training_error(state_dict, reg_result, T_trans):

    X = state_dict.get('all_states')
    Z = state_dict.get('input_data')
    
    X = X[:, T_trans:]
    Z = Z[:, T_trans:]
    T = Z.shape[1]
    d = Z.shape[0]
    
    weights, bias = reg_result
    X_1 = np.concatenate((X, np.ones(shape=(1, T))), axis=0)
    weights_bias = np.concatenate((weights, bias.reshape(1, d)), axis=0)
    Z_hat = weights_bias.T @ X_1
    
    training_error = np.sum(np.linalg.norm(Z_hat-Z, axis=0)**2) / T
        
    return training_error

def av_multi_training_error(state_dicts, reg_result, T_trans):
    errors = []
    
    for i in range(len(state_dicts)):
        errors = errors + [training_error(state_dicts[i], reg_result, T_trans)]
        
    return np.mean(errors)

# Plotting functions

def state_plot(state_dict, plotwith_init, T_trans, node=0):
    X = state_dict.get('all_states')
    if plotwith_init == True:
        state_plot, state_ax = plt.subplots(figsize=(20,5))
        state_ax.plot(X[node][:])
        state_ax.set_title('Plot of States at node {}'.format(node))
        state_ax.set_xlabel('time')
        state_ax.set_ylabel('state of node {}'.format(node))
        state_plot.savefig('state_plot.pdf')
        
        return (np.amin(X[node][:]), np.amax(X[node][:]))
    
    if plotwith_init == False:
        state_plot, state_ax = plt.subplots(figsize=(20,5))
        state_ax.plot(X[node][T_trans:])
        state_ax.set_title('Plot of States at node {}'.format(node))
        state_ax.set_xlabel('time')
        state_ax.set_ylabel('state of node {}'.format(node))
        state_plot.savefig('state_plot.pdf')
    
        return (np.amin(X[node][T_trans:]), np.amax(X[node][T_trans:]))
    
                
def hist_accuracy_plot(actuals, predictions, x_label, y_label, with_bars=False):
    if with_bars == False:
        sns.kdeplot(actuals, label='actual', shade=True, color='red')
        sns.kdeplot(predictions, label='prediction', shade=True, color='skyblue')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
    
    if with_bars == True:
        sns.histplot(actuals, label='actual', color='red', kde=True)
        sns.histplot(predictions, label='prediction', color='skyblue', kde=True)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        
def average_error_traj(prediction_dicts):
    # takes a list of prediction_dict dictionaries and averages the error at each time step across the trajectory
        # prediction_dicts - list of dictionaries prediction_dict
        
    Errors = []
    
    for prediction_dict in prediction_dicts:
        
        z_actuals = prediction_dict.get('z_actuals')
        z_predictions = prediction_dict.get('z_predictions')        
        z_errors = np.array([np.linalg.norm(z_actuals[:, t] - z_predictions[:, t]) for t in range(z_actuals.shape[1])])
        
        Errors = Errors + [z_errors]
    
    Mean_errors = []
    
    for i in range(len(Errors[0])):
        Mean_errors = Mean_errors + [mean([err[i] for err in Errors])]
    
    return np.array(Mean_errors)
    

def prediction_error_plot(z_errors, h, max_lyapunov_exp, plot_with_error_bound = True,
                          log_scale = True, time_cut_off = None, N_transient = 1,
                          title = None, include_title = False):
    # plots the error in prediction over time
        # z_errors - errors to plot
        # lyapunov_spectrum - numpy array with lyapunov exponents; only the largest is used
        # plot_with_error_bound - controls whether the plot contains the predicted error bound
        # log_scale - controls whether the error is plotted on a log scale
        # time_cut_off - controls how long the plot is plotted for
        # N_transient - controls interval length used to fit the multiplicative constant in the error bound
        # system_name - label used in the title of the plot
        # N_traj - number of trajectories used in averaging
        # access - string detailing the number of coordinates access is given to in training
        # include_title - determines whether to include the title for the graph
    
    T_bar = len(z_errors)
    if time_cut_off != None:
        T_bar = min(math.floor(time_cut_off/h), T_bar)
    Time = [i*h*max_lyapunov_exp for i in range(T_bar)]
    z_errors = z_errors[:T_bar]

    if log_scale == True:
        z_errors = np.log(z_errors)
    error_plot, error_ax = plt.subplots(figsize=(20,5))
    error_ax.plot(Time, z_errors, label = 'Actual errors')
    error_ax.set_xlim(0, (time_cut_off - h) * max_lyapunov_exp)
    error_ax.set_ylim(z_errors[1])
    
    if include_title:
        error_ax.set_title(title)
        """
        if N_traj != None:    
            if access == None:
                error_ax.set_title('{} Plot of prediction errors against time averaged over {} trajectories'.format(system_name, N_traj))
            else:
                error_ax.set_title('{} Plot of prediction errors against time averaged over {} trajectories trained with access to {}'.format(system_name, N_traj, access))
        else:
            if access == None:
                error_ax.set_title('{} Plot of prediction errors against time'.format(system_name))
            else:
                error_ax.set_title('{} Plot of prediction errors against time trained with access to {}'.format(system_name, access))
        """
    
    error_ax.set_xlabel('Lyapunov times')
    if log_scale == True:
        error_ax.set_ylabel('log prediction errors')
    else:
        error_ax.set_ylabel('prediction error')
        
    if plot_with_error_bound == True:
        l1 = max_lyapunov_exp 
        
        if log_scale == True:
            z_error_bounds = max(z_errors[:N_transient]) + Time #np.array([t*l1 for t in Time]) # time scale is in Lyapunov times already
            error_ax.set_ylim([min(z_errors) - 5, max(z_errors) + 5])
        if log_scale == False:
            z_error_bounds = max(z_errors[:N_transient]) * np.array([np.exp(t*l1) for t in Time])
            error_ax.set_ylim([0, 2*max(z_errors)])
            
        error_ax.plot(Time, z_error_bounds, label = 'Predicted error bound')

    plt.legend(loc = 'lower right')
    error_plot.savefig('error_plot.pdf')
    error_plot.show()
    
def test_for_esp(training_data, x_0, sd, A, gamma, C, s, zeta, d, N, num_traj):
    # randomly generates num_traj trajectories and averages the distance of these trajectories to a reference trajectory
    # and plots the distance over time.
    x1 = np.array([x_0 for i in range(num_traj)])
    x1 += np.random.uniform(-sd, sd, (num_traj, len(x_0),1))
    print(np.linalg.norm(x1[0,:,0] - x_0[:,0]), 'initial error for 1 trajectory')

    t0 = listening(training_data, x_0, A, gamma, C, s, zeta, d, N)['all_states']
    X1 = []
    for i in range(num_traj):
        X1 = X1 + [listening(training_data, x1[i], A, gamma, C, s, zeta, d, N)['all_states']]
    print(X1)
        
    X1 = np.array(X1)
    print('X1', X1.shape)
    print('t0', t0.shape)
        
    Errors = []
    
    for t1 in X1:
        errors = t1 - t0
        errors = errors.transpose()
        error_norms = [np.linalg.norm(err, 2) for err in errors]
        Errors = Errors + [error_norms]
        
    Errors = np.array(Errors)
    Errors = Errors.transpose()
    
    print('Errors', Errors.shape)
    
    Av_errors = []
    
    for err in Errors:
        Av_errors = Av_errors + [np.mean(err)]
        
    Log_av_errors = Av_errors # np.log(Av_errors)
        
    error_plot, error_ax = plt.subplots(figsize=(20,5))
    error_ax.plot(Log_av_errors)
    error_ax.set_title('Plot of distance between trajectories')
    error_ax.set_xlabel('time')
    error_ax.set_ylabel('average distance between trajectories')
    
    return Av_errors





















