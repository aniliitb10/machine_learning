from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable


def load_restaurant_data() -> Tuple[np.ndarray[float], np.ndarray[float]]:
    """ Just a helper method to load the restaurant data """
    data = np.loadtxt("data/restaurant_data.txt", delimiter=',')
    x = data[:, 0]
    y = data[:, 1]
    return x, y


def bold(string):
    return f'\033[1m{string}\033[0;0m'


def print_table(rows=None, **kwargs):
    """ A function to print @rows rows columns like a table """
    table = PrettyTable()

    for column_name, column in kwargs.items():
        table.add_column(bold(column_name),
                         [f'{int(data)}' if data.is_integer() else f'{data:.3f}'  # float formatting
                          for data in column[:rows or len(column)]])

    print(table)


def plot_cost_history(cost_history, split_index):
    """ A helper function to help visualize cost history"""
    cost_l, cost_r = cost_history[:split_index], cost_history[split_index:]
    x_l, x_r = np.arange(split_index), np.arange(split_index, len(cost_history))

    # plotting cost vs iteration
    fig = plt.figure(figsize=(10, 6))
    plt.subplots_adjust(hspace=0.5)

    ax_main = plt.subplot(211)
    ax_main.plot(np.concatenate([x_l, x_r]), np.concatenate([cost_l, cost_r]))
    ax_main.set_xlabel(f'Iterations [0, {len(cost_history)})')

    ax_bottom_left = plt.subplot(223)
    ax_bottom_left.plot(x_l, cost_l)
    ax_bottom_left.set_xlabel(f'Iterations [0, {split_index})')

    ax_bottom_right = plt.subplot(224)
    ax_bottom_right.plot(x_r, cost_r)
    ax_bottom_right.set_xlabel(f'Iterations [{split_index}, {len(cost_history)})')

    # some decoration
    fig.suptitle('Cost vs Iteration', fontweight="bold")
    fig.supylabel('Cost', fontweight="bold")
    plt.show()


def print_training_data(x_train: np.ndarray, y_train: np.ndarray, rows=None):
    n = x_train.shape[1]
    x_columns: List[str] = [f'x{i}s' for i in range(n)]
    xs_map = {x_columns[i]: x_train[:, i] for i in range(n)}

    print_table(**xs_map, y_train=y_train, rows=rows)


def print_intermediate_details(x_train: np.ndarray, y_train: np.ndarray, w_initial=None, b_initial=0):
    m, n = x_train.shape
    w_initial = np.ones(n) if w_initial is None else w_initial
    b_initial = 0 if b_initial is None else b_initial

    f_wb = np.dot(x_train, w_initial) + b_initial
    error = f_wb - y_train

    x_columns: List[str] = [f'x{i}s' for i in range(n)]
    xs_map = {x_columns[i]: x_train[:, i] for i in range(n)}

    print_table(**xs_map, f_wb=f_wb, y_train=y_train, error=error)

    print('\n\n')  # maintain some distance

    error_xj = np.dot(x_train.T, error).reshape((1, -1))
    error_xj_columns: List[str] = [f'sum(error * x{i}s)' for i in range(n)]
    error_xj_map = {error_xj_columns[i]: error_xj[:, i] for i in range(n)}
    print_table(**error_xj_map)

    print('\n\n')  # maintain some distance

    dj_dw = error_xj / m
    dj_dw_columns: List[str] = [f'dj_dw{i}' for i in range(n)]
    dj_dw_map = {dj_dw_columns[i]: dj_dw[:, i] for i in range(n)}
    print_table(**dj_dw_map)
