import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models
from dataclasses import dataclass
#ML libs
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor

@dataclass
class ModelTrainingConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainingConfig()

    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test=(
                    train_arr[:,:-1],
                    train_arr[:,-1],
                    test_arr[:,:-1],
                    test_arr[:,-1]
            )
            models={
                "LinearRegression":LinearRegression(),
                "Lasso":Lasso(),
                "Ridge":Ridge(),
                "K-Neighbours Regressor":KNeighborsRegressor(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "RandomForestRegressor":RandomForestRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBoostRegressor":CatBoostRegressor(logging_level='Silent'),
                "AdaBoostRegressor":AdaBoostRegressor(),
                "GradientBoostRegressor":GradientBoostingRegressor()
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")

            logging.info("Best model found on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted= best_model.predict(X_test)
            r2=r2_score(y_test,predicted)
            return r2

        except Exception as e:
            raise CustomException(e,sys)
            