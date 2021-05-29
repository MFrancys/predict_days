import numpy as np

def transform_tuple_target(data, event_target, time_target):
    """Include the time factor in the target variable"""
    tuple_target = [
        (row[event_target], row[time_target]) for index, row in data.iterrows()
    ]
    data_tuple_target = np.array(tuple_target, dtype=[('sold', "bool"), ('DI', 'float')])
    return data_tuple_target

