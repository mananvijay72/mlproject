import os 
import sys
from src.exception import CustomException
from src.logger import logging
import dill
import pandas as pd
import numpy as np


def save_object(file_path, obj):
    '''
    Function to save the object to a given file path
    '''

    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)