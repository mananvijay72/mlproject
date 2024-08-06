import os 
import sys
from src.exception import CustomException
from src.logger import logging
import dill
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

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
    

def evaluate_model(X_train, y_train, X_test, y_test, models:dict, params):

    '''
    Returns the metrics of all the given models 
    '''
    try:
        report = {}

        for i in range(len(models)):

            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            if model_name in params.keys():
                param = params[model_name]
            else:
                param = {}

            #Training Model using grid search CV
            
            grid = GridSearchCV(model, param_grid=param, cv=3)
            grid.fit(X_train, y_train)

            model.set_params(**grid.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            #calculating model metrics

            r2_train = r2_score(y_train, y_train_pred)
            r2_test = r2_score(y_test, y_test_pred)

            #storing r2 score in report dictionary

            report[list(models.keys())[i]] = r2_test
        
        return report
    
    except Exception as e:
        raise CustomException(e,sys)


