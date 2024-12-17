This repository takes the code from the May submission as base. Thereafter, I am working on cleaning up this code, correcting errors and refining it for its purpose. This constitutes the main branch.

The May submission branch contains the final code used in the May submission of our paper. It is used as the basis for the repository where we refine this after the May submission. It should not be pushed back into the main branch.

The Checking regression covariance function branch was branched out from the main branch during the code cleaning process on 30 May 2024, Thursday. In this branch I added a whole lot of code trying to pinpoint the discrepancy between the readouts generated using the cv code and the normal code used for graphing. As far as I can tell while all variables and functions are the same, the only difference is that in the one the readout was trained on data stored as a numpy.array inside a dictionary, whereas in the other the training data was stored directly as a numpy.array, not inside a dictionary. Because of the large nature of the matrices, this led to differences in computation during the np.linalg.solve function. What is concerning is that this solve function is not so accurate with such a big matrix. How relevant is all the maths we have done then? Nevertheless, it does seem we are able to learn the dynamical system. My plan is to simply standardize the two codes so they agree (albeit both are wrong, though I'm not sure which method will be more accurate or even how accurate the different methods are), and then run the cv code to tune the hyperparameters and see where we go from there.

The code refining for December review was brnached out from Main on 13 December 2024 Friday. In this code I will work on new graphs and figures for the revision of the paper. This code should not be merged back into main unless at the end I am satisfied with it.