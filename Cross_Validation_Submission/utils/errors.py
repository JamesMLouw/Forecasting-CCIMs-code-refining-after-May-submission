import numpy as np

def calculate_mse(y_true, y_pred, shift=None, scale=None):

    """
    Calculate Mean Squared Error (MSE) between true and predicted values.
    If shift and scale are not None, then unshifts and unscales the data. 

    Parameters
    ----------
    y_true : array_like
        Numpy array of true target values.
    y_pred : array_like
        Numpy array of predicted target values.
    shift : float
        The shift that was implemented in the normalisation process
    scale : float
        The scale that was implemented in the normalisation process

    Returns
    -------
    mse : float
        Mean Squared Error.
    """
    
    if shift is not None and scale is not None:
        # Destandardize the data if required
        y_true = y_true * (1/scale) + shift
        y_pred = y_pred * (1/scale) + shift

    # Calculate MSE
    # mse = np.mean((y_true - y_pred)**2) This is not quite the true mse, I think
    errors = y_true - y_pred
    mse = np.mean([np.linalg.norm(a)**2 for a in errors])
    
    # we could also use mse = np.mean((y_true - y_pred)**2) * y_true.shape[1]

    return mse

