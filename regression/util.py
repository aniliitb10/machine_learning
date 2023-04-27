import numpy as np
from typing import Tuple, List
from prettytable import PrettyTable


def rel_change(new_num: float, old_num: float) -> float:
    """
    Like percent change but without multiplying by 100
    :param new_num: latest number
    :param old_num: the number used as base for this calculation
    :return: relative change
    """
    return (new_num - old_num) / old_num

def load_restaurant_data() -> Tuple[np.ndarray[float], np.ndarray[float]]:
    """ Just a helper method to load the restaurant data """
    data = np.loadtxt("data/restaurant_data.txt", delimiter=',')
    X = data[:,0]
    y = data[:,1]
    return X, y


def print_table(rows=None, decimal_points=5, **kwargs):
    """ A function to print @rows rows columns like a table """
    table = PrettyTable()
    
    for column_name, column in kwargs.items():
        table.add_column(column_name, [f'{data:.6f}' for data in column[:rows or len(column)]]) # control float formatting
    
    print(table)
