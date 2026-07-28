import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utils import save_object

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

@dataclass
class DataTransformerConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.dataTransformerConfig=DataTransformerConfig()

    def get_data_transformer_obj(self):
        try:
            num_columns=["writing_score","reading_score"]
            cat_columns=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            numerical_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            logging.info("Numerical columns scaling completed")
            categorical_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    # ("scaler",StandardScaler())
                ]
            )
            logging.info("Categorical columns encoding completed")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",numerical_pipeline,num_columns),
                ("cat_pipeline",categorical_pipeline,cat_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transoformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessor_obj=self.get_data_transformer_obj()

            target_column_name="math_score"

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframe")

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)   
            ]

            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)   
            ]

            logging.info("Saved preprocessing object")

            save_object(
                file_path=self.dataTransformerConfig.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            return(
                train_arr,
                test_arr,
                self.dataTransformerConfig.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)


