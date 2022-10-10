import numpy as np
from typing import Tuple
from regression import util


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

    # Each element of x (i.e. data) is to be multiplied by error of that row / example data
    # and then column-wise sum represents the error in that column => error in that weight
    # To achieve this:
    #  1) Take dot product of transpose of x by error
    #  2) Column-wise sum is automatically taken care of
    dj_dw = np.dot(x.T, error)
    return dj_dw / x.shape[0], np.sum(error) / x.shape[0]  # x.shape[0] is m


def gradient_descent(x: np.ndarray, y: np.ndarray, w_in: np.ndarray, b_in, alpha: float, num_iters: int,
                     change_thr: float = 1.e-7)\
        -> Tuple[np.ndarray, float]:
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
      change_thr (float)  : if relative change between two consecutive costs is less than change_thr, then return
    Returns:
      w (ndarray (n,)) : Updated values of parameters
      b (scalar)       : Updated value of parameter
      """
    w = w_in
    b = b_in

    dj_dw, dj_db = compute_gradient(x, y, w, b)
    w = w - alpha * dj_dw
    b = b - alpha * dj_db
    old_cost, new_cost = 0, compute_cost(x, y, w, b)

    for i in range(num_iters-1):
        old_cost = new_cost

        # actual calculation
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        new_cost = compute_cost(x, y, w, b)

        rel_change = util.rel_change(new_cost, old_cost)
        if abs(rel_change) <= change_thr:
            return w, b

    return w, b
