import numpy as np
from typing import Tuple


def compute_cost(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    """
    compute cost
    Args:
      x (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter

    Returns:
      cost (scalar): cost
    """
    f_w_b: np.ndarray = np.dot(x, w) + b  # it contains w.x + b for all m examples
    error = f_w_b - y  # errors for all m examples
    return np.sum(np.dot(error, error)) / (2 * x.shape[0])  # x.shape[0] is m


def compute_gradient(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> Tuple[np.ndarray, float]:
    """
    Computes the gradient for linear regression
    Args:
      x (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter

    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b.
    """
    f_w_b: np.ndarray = np.dot(x, w) + b  # f_w_b for all m examples
    error = f_w_b - y  # error for all m examples

    # Each element of x (i.e. data) is to be multiplied by error of that row
    # and then columns wise sum represents the error in that column => error in that weight
    # To achieve this:
    #  1) Multiply transpose of x by (broadcasted) error
    #  2) Take transpose of the product => Each element is a product of original element and error of that row
    #  3) Take column wise sum => each such sum represents the error in that column => weight
    dj_dw = np.multiply(x.T, error).T.sum(axis=0)
    return dj_dw / x.shape[0], np.sum(error) / x.shape[0]  # x.shape[0] is m


def gradient_descent(x: np.ndarray, y: np.ndarray, w_in: np.ndarray, b_in, alpha: float, num_iters: int) -> \
        Tuple[np.ndarray, float]:
    """
    Performs batch gradient descent to learn theta. Updates theta by taking
    num_iters gradient steps with learning rate alpha

    Args:
      x (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters
      b_in (scalar)       : initial model parameter
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent

    Returns:
      w (ndarray (n,)) : Updated values of parameters
      b (scalar)       : Updated value of parameter
      """
    w = w_in
    b = b_in
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

    return w, b
