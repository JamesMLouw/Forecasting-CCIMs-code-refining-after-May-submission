
### ESN class code

import numpy as np
import estimators.ESN_helperfunctions as esn_help
from time import process_time


class ESN:
    
    """
    ESN object
    
    Attributes
    ----------
    ld : float
        Reguralisation used for Tikhonov least squares regression
    gamma : float
        Coefficient of input mask C
    spec_rad : float
        Spectral radius of connectivity matrix A
    s = 0
        Coefficient of input shift matrix zeta
    washout : int
        Amount of washout to use during training
    N : int
        Dimension of state space
    d_in : int
        Dimension of input space
    d_out : int
        Dimension of output space
    zeta : numpy_array of floats; shape (N,1)
        Input shift matrix
    C : numpy_array of floats; shape (N,d_in)
        Input mask        
    A : numpy_array of floats; shape (N,N)
        Connectivity matrix
    x_0 : numpy_array of floats; shape (N,1)
        Initial state space point
    W : numpy_array of floats; shape (d_out,N)
        Readout matrix
    bias : numpy_array of floats; shape (d_out, 1)
        Output shift vector
        
    Methods
    -------
    Train(training_input, training_teacher)
        Performs training using the training input against the training teacher for the ESN
    Forecast(testing_input)
        Performs testing using a new set of testing input 
    PathContinue(latest_input, nhorizon)
        Simulates forward in time using the latest input for nhorizon period of time
        This requires a training where the training_teacher = training_input, so that the ESN can be run autonomously
    """
    
    def __init__(self, ld, gamma, spec_rad, s, N, d_in, d_out, washout):
        
        
        # Instance attributes that are user defined
        self.ld = ld
        self.gamma = gamma
        self.spec_rad = spec_rad
        self.s = s
        self.N = N
        self.d_in = d_in
        self.d_out = d_out
        self.washout = washout    
        
        # Instance attributes that are randomly generated
        self.zeta = esn_help.gen_matrix(shape=(N,1), density=1, pdf="ones", seeded=False)
        np.random.seed(0)
        self.C = esn_help.gen_matrix(shape=(N,d_in), density=1, sd=2, mean=-1, pdf="uniform", seeded=False)
        self.A = esn_help.gen_matrix(shape=(N,N), density=0.01, sd=2, mean=-1, pdf="uniform", seeded=False)
        self.A = esn_help.spectral_radius_matrix(self.A, spec_rad)
        self.x_0 = np.zeros(shape=(N,1), dtype=float)
        self.W = np.zeros(shape=(N, d_out), dtype=float)
        self.bias = np.zeros(shape=(d_out,1))

        # Instance attributes storing arrays created by methods
        self.training_input = None      # Stores training input seen during training
        self.x_start_path_continue = np.zeros(shape=(N,1)) # Stores last state after training
        self.x_start_forecasting = np.zeros(shape=(N,1)) # Stores next state after training
        
        # Instance attributes storing data dependent values created by methods
        self.ninputs = None             # Store training input length
        self.ntargets = None            # Store number of targets output in testing
        self.nhorizon = None            # Store length of forecasting horizon
        
       
    def Train(self, training_input, training_teacher = None):
        
        """
        Performs training using the training input against the training teacher
        After a washout period, the ESN readout is learnt by a regression
        
        Parameters
        ----------
        training_input : array_like
            Training input for training the ESN. Must have format (nsamples, ndim)
        training_teacher : array_like
            Training teacher for training the ESN. Must have format (nsamples, ndim)

        Returns
        -------
        ESN : class_instance
            ESN object with training attributes initialised
        
        Note, for the PathContinue and Forecast tasks, this training only fits the ESN
        to perform these tasks on a trajectory continuing directly on the one on which it
        was trained
        """
        print('difference teacher innput', np.max(np.abs(training_input - training_teacher)))
        
        # Assign as instance attribute that are based on training input data
        self.training_input = training_input 
        self.ninputs = training_input.shape[0]
        
        if self.d_in != training_input.shape[1]:
            raise ValueError("The dimension of the training input is incorrect")
        
        # Assign instance attributes that are related to the training teacher data
        self.ntargets = training_teacher.shape[1]
        
        if self.d_out != training_teacher.shape[1]:
            raise ValueError("The dimension of the training teacher is incorrect")
        
        # Check that the training input and training teacher sizes are the same
        nteacher = training_teacher.shape[0]
        if self.ninputs != nteacher:
            raise ValueError("The size of the training teacher and training inputs do not match")
        
        # Check that the washout is not greater than the size of the inputs
        if self.washout >= self.ninputs:
            raise ValueError("The washout is too large")
                    
        state_dict = esn_help.listening(self.training_input, self.x_0, self.A, self.gamma, 
                                        self.C, self.s, self.zeta, self.d_in, self.N)
        print('difference input state dict input', np.max(np.abs(targets - state_dict['input_data'])))
        
        #state_dict['input_data'] = targets
        if training_teacher == None:
            reg_result = esn_help.regression_covariance(self.ld, state_dict, self.washout)
        else:
            targets = training_teacher.transpose()    
            reg_result = esn_help.regression_covariance_targets(self.ld, state_dict['all_states'],
                                                            targets, self.washout)
            
        reg_result = esn_help.regression_covariance(self.ld, state_dict, self.washout)
        reg_result1 = esn_help.regression_covariance_targets(self.ld, state_dict['all_states'],
                                                            targets, self.washout)
        reg_result3 = esn_help.regression_covariance_targets(self.ld, state_dict['all_states'],
                                                            targets, self.washout)
        reg_result2 = esn_help.regression_covariance_targets(self.ld, state_dict['all_states'],
                                                            state_dict['input_data'], self.washout)
        print('difference reg results: different functions same variables', np.max(np.abs(reg_result[0] - reg_result2[0])))
        print('difference reg results: same function different variables', np.max(np.abs(reg_result2[0] - reg_result1[0])))
        # this is very strange, as far as i can tell the inputs for these two are exactly equal numerically, in type and shape and the functions are the same, yet when calculated the functions output different values. The problem must be somewhere in numpy I think.
        # in any case, the function line inputted into reg_result is in agreement with the results from the graphing side, so I think we ought to use this one for training.
        print('input data', targets.shape, type(targets))
        print('stat dict', state_dict['input_data'].shape, type(state_dict['input_data']))
        print(all([all(x) for x in state_dict['input_data'] == targets]))
        print('stat dict' , state_dict['input_data'])
        print('targets',targets)

        self.W = reg_result[0]
        self.bias = reg_result[1]
        #print('W shape:',self.W.shape)
        #print('bias shape:',self.bias.shape)
        self.x_start_path_continue = state_dict['last_state'] # x_-1
        self.x_start_forecasting = esn_help.state(self.x_start_path_continue,
                                                  training_input[-1].reshape((self.d_in,1)),
                                                  self.A, self.gamma, self.C, self.s,
                                                  self.zeta, self.d_in) # x_0
                    
        return self
    
    
    def Forecast(self, testing_input):
        
        """
        For some testing input, use the trained ESN object to generate output based on the 
        training teacher that was given
        
        Parameters
        ----------
        testing_input : array_like
            New input given that should be used for forecasting. Must have format (nsamples, ndim)

        Returns
        -------
        output : array_like
            ESN forecasts, will be of the same type as the training teacher. Will have format (nsamples, ndim)
        
        Note
        ----
        testing_input has to continue directly on from training_input; 
        """
        
        # Assign as instance attribute testing input related 
        self.nhorizon = testing_input.shape[0]
        
        # Initialise store for the forecast output
        output = np.zeros((self.nhorizon, self.d_out))
        x_curr = self.x_start_forecasting # think of this as x_0
        
        # Iterate through the testing horizon
        for t in range(self.nhorizon):
            z_curr = testing_input[t].reshape((self.d_in,1)) # testing_input[0] is z_0
            x_curr = esn_help.state(x_curr, z_curr, self.A, self.gamma, self.C, self.s, self.zeta, self.d_in)
            output[t] = esn_help.observation(x_curr, (self.W, self.bias)) # self.W @ x_curr + self.bias
        
        return output
    
    
    def PathContinue(self, latest_input, nhorizon):
        
        """
        Simulates forward in time using the latest input for nhorizon period of time
        
        Parameters
        ----------
        latest_input : array_like
            Starting input to path continue from (think of this as at time -1)
        nhorizon : int
            Period of time to path continue over

        Returns
        -------
        output : array_like
            Output of forecasting. Will have format (nsamples, ndim)
            
        Note
        ----
        latest_input has to be the last input from the training trajectory
        """
        
        # Assign as instance attribute the testing horizon given
        self.nhorizon = nhorizon
        
        # Initialise store for the forecast output
        output = np.zeros((self.nhorizon, self.d_out, 1))
        z_curr = latest_input.reshape((self.d_in,1)) # z_-1
        x_curr = self.x_start_path_continue # x_-1
                
        # Iterate through the testing horizon
        for t in range(self.nhorizon):
            x_curr = esn_help.state(x_curr, z_curr, self.A, self.gamma, self.C, self.s, self.zeta, self.d_in)
            z_curr = esn_help.observation(x_curr, (self.W, self.bias)) # self.W @ x_curr + self.bias
            output[t] = z_curr
                
        return output


