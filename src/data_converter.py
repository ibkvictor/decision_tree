import numpy as np
import pandas as pd

def dataframe_to_dict(data):
    """
        converts dataframe (data wrapper) to dict (native python data format)
        argument: 
            data: (dataframe object)
        return: 
            dict: contains data in lists
    """
    return {col: extract(data[col]) for col in list(data.columns)}

def extract(data):
    """
        converts pandas series (data wrapper) to list (native python data format)
            argument: 
                data: (pandas series object)
            return:
                list: contains data items.
    """
    return [i for i in data]