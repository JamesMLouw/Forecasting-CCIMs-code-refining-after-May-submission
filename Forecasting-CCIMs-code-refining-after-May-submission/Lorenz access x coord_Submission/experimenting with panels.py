#%%
import matplotlib.pyplot as plt
import numpy as np
import math

# Get the angles from 0 to 2 pie (360 degree) in narray object
X = np.arange(0, math.pi*2, 0.05)

# Using built-in trigonometric function we can directly plot
# the given cosine wave for the given angles
Y1 = np.sin(X)
Y2 = np.cos(X)
Y3 = np.tan(X)
Y4 = np.tanh(X)

# Initialise the subplot function using number of rows and columns
figure, (a1,a2) = plt.subplots(2, 2)

# For Sine Function
a1[0].plot(X, Y1)
a2[0].plot(X,Y2)
# Combine all the operations and display
plt.show()
# %%
