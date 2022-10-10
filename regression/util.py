def rel_change(new_num: float, old_num: float) -> float:
    """
    Like percent change but without multiplying by 100
    :param new_num: latest number
    :param old_num: the number used as base for this calculation
    :return: relative change
    """
    return (new_num - old_num) / old_num
