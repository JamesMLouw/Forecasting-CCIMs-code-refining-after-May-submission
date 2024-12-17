
### NGRC Code written as a class - adapted from original NGRC paper's codes at https://github.com/quantinfo/ng-rc-paper-code

import numpy as np
from itertools import combinations_with_replacement
from math import comb


class NGRC:
    
    """
    Next-generation reservoir computing (NGRC) object
    
    Attributes
    ----------
    ndelay : int
        Numer of delay terms to use
    deg : int
        Degree of monomials chosen
    reg : float
        Reguralisation used for Tikhonov least squares regression
    washout : int
        Amount of washout to use during training
        
    Methods
    -------
    Train(training_input, training_teacher)
        Performs training using the training input against the training teacher in the NGRC method
    PathContinue(latest_input, nhorizon)
        Simulates forward in time using the latest input for nhorizon period of time
    """
    
    def __init__(self, ndelay, deg, reg, washout):
        
        # Instance attributes that are user defined
        self.ndelay = ndelay        # Number of delay terms to use         
        self.deg = deg              # Degree of monomials chosen 
        self.reg = reg              # Regularisation for least squares Tikhonov
        self.washout = washout      # Washout for training data
        
        # Instance attributes storing arrays generated by methods
        self.Xtrain = None          # Store for linear feature vectors seen during training
        self.Otrain = None          # Store for all feature vectors seen during training
        self.Wout = None            # Weights computed from regression
        
        # Instance attributes storing data dependent values generated by methods
        self.ninputs = None         # Training size derived from length of data seen during training (input size+1)
        self.nhorizon = None        # Stores the forecasting horizon given 
        self.nfeatures = None       # Stores the number of features given into the system  
        self.dlin = None            # Stores the number of NGRC linear features (including delays)
        self.dnonlin = None         # Stores the number of NGRC nonlinear features 
        self.dtot = None          # Stores the number of NGRC total features
        
        
    
    def Train(self, training_input, training_teacher):
        
        """
        Performs training using the training input against the training teacher in the NGRC method
        
        Parameters
        ----------
        training_input : array_like
            Training input for training in NGRC. Must have format (nsamples, ndim)
        training_teacher : array_like
            Training teacher for training in NGRC. Must have format (nsamples, ndim)

        Returns
        -------
        NGRC : class_instance
            NGRC object with training attributes initialised
        """
        
        # Define size and dimension of training data
        self.ninputs = training_input.shape[0] 
        # Define dimension of input based on training data
        self.nfeatures = training_input.shape[1]
        
        # Size of linear part of feature vector
        self.dlin = self.ndelay * self.nfeatures
        
        # Size of nonlinear part of feature vector
        self.dnonlin = 0
        for inter_deg in range(2, self.deg+1):
            self.dnonlin = self.dnonlin + comb(inter_deg+self.dlin-1, self.dlin-1)
        
        # Total size of feature vector: constant + linear + nonlinear
        self.dtot = 1 + self.dlin + self.dnonlin
            
        # Create array to hold linear part of feature vector
        Xtrain = np.zeros((self.dlin, self.ninputs))
        
        # Fill in the linear part of the feature vector
        for delay in range(self.ndelay):
            for j in range(delay, self.ninputs):
                Xtrain[self.nfeatures*delay:self.nfeatures*(delay+1), j] = training_input[j-delay, :]
        
        # Create feature vector over training time
        Otrain = np.ones((self.dtot, self.ninputs-self.washout))
        
        # Copy over linear part (shifting by one to account for constant)
        Otrain[1:self.dlin+1, :] = Xtrain[:, self.washout:self.ninputs]
        
        # Fill in nonlinear part of the feature vector
        Orow = 1 + self.dlin
        # Iterate through each monomial degree
        for inter_deg in range(2, self.deg+1):
            # Generate iterator of combinations rows of X for each degree
            iter_monomials = combinations_with_replacement(range(self.dlin), inter_deg)
            # Fill up the rows of O train for each monomial 
            for X_row_ids in iter_monomials:
                monomial_row = Xtrain[X_row_ids[0], self.washout:self.ninputs]
                for row_id in range(1, inter_deg):
                    monomial_row = monomial_row * Xtrain[X_row_ids[row_id], self.washout:self.ninputs]
                Otrain[Orow] = monomial_row
                Orow = Orow + 1
        
        # Assign as instance attributes the linear and nonlinear feature vectors
        self.Xtrain = Xtrain
        self.Otrain = Otrain
        
        # Ridge regression train W_out with X_i+1 - X_i
        self.Wout = ((training_teacher[self.washout:] - training_input[self.washout:]).T @ Otrain.T @ np.linalg.pinv(Otrain @ Otrain.T + self.reg * np.identity(self.dtot))).T
        
        return self
    
    
    
    def PathContinue(self, latest_input, nhorizon):
        
        """
        Simulates forward in time using the latest input for nhorizon period of time
        
        Parameters
        ----------
        latest_input : array_like
            Starting input to path continue from
        nhorizon : int
            Period of time to path continue over

        Returns
        -------
        output : array_like
            Output of forecasting
        """
        
        # Assign as instance attributes the testing horizon given
        self.nhorizon = nhorizon
        
        # Create store for feature vectors for prediction
        Otest = np.ones(self.dtot)              # full feature vector
        Xtest = np.zeros((self.dlin, self.nhorizon+1))    # linear portion of feature vector
    
        # Fill in the linear part of the feature vector with the latest input data and delay
        Xtest[0:self.nfeatures, 0] = latest_input
        Xtest[self.nfeatures: , 0] = self.Xtrain[0:self.dlin-self.nfeatures, -1]
        
        # Apply W_out to feature vector to perform prediction
        for j in range(self.nhorizon):
            # Copy linear part into whole feature vector
            Otest[1:self.dlin+1] = Xtest[:, j] # shift by one for constant
        
            # Fill in the nonlinear part
            Orow = 1 + self.dlin
            # Iterate through each monomial degree
            for inter_deg in range(2, self.deg+1):
                # Generate iterator of combinations rows of X for each degree
                iter_monomials = combinations_with_replacement(range(self.dlin), inter_deg)
                # Fill up the rows of O test for each monomial 
                for X_row_ids in iter_monomials:
                    monomial_row = Xtest[X_row_ids[0], j]
                    for row_id in range(1, inter_deg):
                        monomial_row = monomial_row * Xtest[X_row_ids[row_id], j]
                    Otest[Orow] = monomial_row
                    Orow = Orow + 1
                    
            # Fill in the delay taps of the next state
            Xtest[self.nfeatures:self.dlin, j+1] = Xtest[0:self.dlin-self.nfeatures, j]
            # Perform a prediction
            Xtest[0:self.nfeatures, j+1] = Xtest[0:self.nfeatures, j] + Otest @ self.Wout
            
        # Store the forecasts as output
        output = Xtest[0:self.nfeatures, 1:].T
        
        return output
    
    
