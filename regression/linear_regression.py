from typing import Tuple, List

import numpy as np


def compute_cost(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lambda_: float = 0) -> float:
    """
    compute cost
    Args:
      x (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter
      lambda_ (scalar) : model parameter

    Returns:
      cost (scalar): cost
    """
    m = x.shape[0]
    f_w_b: np.ndarray = np.dot(x, w) + b  # it contains w.x + b for all m examples
    error = f_w_b - y  # errors for all m examples
    return (np.dot(error, error) + lambda_ * np.dot(w, w)) / (2 * m)


def compute_gradient(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lambda_: float = 0) -> \
        Tuple[np.ndarray, float]:
    """
    Computes the gradient for linear regression
    Args:
      x (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter
      lambda_ (scalar) : model parameter

    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b.
    """
    m = x.shape[0]
    f_w_b: np.ndarray = np.dot(x, w) + b  # f_w_b for all m examples
    error = f_w_b - y  # error for all m examples

    # Each element of x (i.e. data) is to be multiplied by error of that row / example data
    # and then column-wise sum represents the error in that column => error in that weight
    # To achieve this:
    #  1) Take dot product of transpose of x by error
    #  2) Column-wise sum is automatically taken care of
    dj_dw = np.dot(x.T, error)
    return (dj_dw + lambda_ * w) / m, np.sum(error) / m


def gradient_descent(x: np.ndarray, y: np.ndarray, w_in: np.ndarray, b_in, alpha: float, lambda_: float = 0,
                     num_iters: int = 10000, with_history: bool = False) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Performs batch gradient descent to learn theta. Updates theta by taking
    num_iters gradient steps with learning rate alpha

    Args:
      x (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters
      b_in (scalar)       : initial model parameter
      alpha (float)       : Learning rate
      lambda_ (scalar)    : model parameter
      num_iters (int)     : number of iterations to run gradient descent
      with_history(bool)  : if True, then returns the const history as well
    Returns:
      w (ndarray (n,))   : Updated values of parameters
      b (scalar)         : Updated value of parameter
      cost(ndarray (n, 3): List of tuple of w,b and cost if @with_history is True, else an empty array. False by default
      """
    w = w_in
    b = b_in
    cost_history: List[float] = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b, lambda_)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if with_history:
            cost_history.append(compute_cost(x, y, w, b, lambda_))

    return w, b, np.array(cost_history)
