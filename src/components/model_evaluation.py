from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from sklearn.metrics import f1_score
from src.exception import MyException
from src.constants import TARGET_COLUMN
from src.logger import logging
from src.utils.main_utils import load_object
import sys
import pandas as pd
from typing import Optional
from src.entity.s3_estimator import Proj1Estimator
from dataclasses import dataclass

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact, schema_config: dict):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self._schema_config = schema_config
        except Exception as e:
            raise MyException(e, sys) from e

    def get_best_model(self) -> Optional[Proj1Estimator]:
        """
        Method Name :   get_best_model
        Description :   This function is used to get model from production stage.
        
        Output      :   Returns model object if available in s3 storage
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path=self.model_eval_config.s3_model_key_path
            proj1_estimator = Proj1Estimator(bucket_name=bucket_name,
                                               model_path=model_path)

            if proj1_estimator.is_model_present(model_path=model_path):
                return proj1_estimator
            return None
        except Exception as e:
            raise  MyException(e,sys)
        
#    def _map_gender_column(self, df):
#        """Map Gender column to 0 for Female and 1 for Male."""
#        logging.info("Mapping 'Gender' column to binary values")
#        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype(int)
#        return df

#    def _create_dummy_columns(self, df):
#        """Create dummy variables for categorical features."""
#        logging.info("Creating dummy variables for categorical features")
#        df = pd.get_dummies(df, drop_first=True)
#        return df

#    def _rename_columns(self, df):
#        """Rename specific columns and ensure integer types for dummy columns."""
#        logging.info("Renaming specific columns and casting to int")
#        df = df.rename(columns={
#            "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
#            "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
#        })
#        for col in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
#            if col in df.columns:
#                df[col] = df[col].astype('int')
#        return df
    
#    def _drop_id_column(self, df):
#        """Drop the 'id' column if it exists."""
#        logging.info("Dropping 'id' column")
#        if "_id" in df.columns:
#            df = df.drop("_id", axis=1)
#        return df


    def _map_default_on_file_column(self, df):
        """Map 'cb_person_default_on_file' column to binary values."""
        logging.info("Mapping 'cb_person_default_on_file' column to binary values")
        df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'Y': 1, 'N': 0}).astype(int)
        return df

    def _cap_employee_length(self, df):
        """Cap specified columns to their defined maximum values."""
        logging.info("Capping columns based on schema config")

        cap_config = self._schema_config.get("cap_columns", {})
        for col, cap_value in cap_config.items():
            if col in df.columns:
                num_capped = (df[col] > cap_value).sum()
                df[col] = df[col].apply(lambda x: cap_value if pd.notnull(x) and x > cap_value else x)
                logging.info(f"Capped {num_capped} values in '{col}' to {cap_value}")
            else:
                logging.warning(f"Column '{col}' not found in DataFrame")

        return df


    def _fill_missing_values(self, df):
        """Fill missing values with median for specified columns."""
        logging.info("Filling missing values with median")

        for item in self._schema_config.get("impute_columns", []):
            for col, strategy in item.items():
                if col not in df.columns:
                    logging.warning(f"Column '{col}' not found in DataFrame")
                    continue

                if strategy != "median":
                    logging.warning(f"Unsupported strategy '{strategy}' for column '{col}' — only 'median' is supported")
                    continue

                median_value = df[col].median()
                missing_count = df[col].isnull().sum()
                df[col] = df[col].fillna(median_value)
                logging.info(f"Filled {missing_count} missing values in '{col}' with median: {median_value}")

        return df


    def _one_hot_encode_columns(self, df):
        """Apply one-hot encoding to specified categorical columns."""
        logging.info("Applying one-hot encoding to selected columns")

        one_hot_cols = self._schema_config.get("one_hot_columns", [])

        for col in one_hot_cols:
            if col in df.columns:
                logging.info(f"Encoding column: {col}")
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            else:
                logging.warning(f"Column '{col}' not found in DataFrame")

        return df
    

    def _ordinal_encode_columns(self, df):
        """Apply ordinal encoding to specified columns based on schema config."""
        logging.info("Applying ordinal encoding to selected columns")

        ordinal_config = self._schema_config.get("ordinal_columns", {})

        for col, config in ordinal_config.items():
            if col in df.columns:
                categories = config.get("categories", [])
                logging.info(f"Encoding column '{col}' with categories: {categories}")
                df[col] = pd.Categorical(df[col], categories=categories, ordered=True).codes
            else:
                logging.warning(f"Column '{col}' not found in DataFrame")

        return df

    def _drop_id_column(self, df):
        """Drop the 'id' column if it exists."""
        logging.info("Dropping 'id' column")
        drop_col = self._schema_config['drop_columns']
        if drop_col in df.columns:
            df = df.drop(drop_col, axis=1)
        return df

    def _convert_boolean_columns(self, df):
        """Convert 'T'/'F' categorical values to binary integers (1/0)."""
        logging.info("Converting 'True'/'False' categorical values to binary integers")

        boolean_cols = self._schema_config.get("boolean_columns", [])

        for col in boolean_cols:
            if col in df.columns:
                logging.info(f"Unique values before mapping in '{col}': {df[col].unique()}")
                df[col] = df[col].map({'True': 1, 'False': 0}).astype(int)
                logging.info(f"Converted column '{col}' from 'True'/'False' to 1/0")
            else:
                logging.warning(f"Column '{col}' not found in DataFrame")

        return df



    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate trained model 
                        with production model and choose best model 
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]

            logging.info("Test data loaded and now transforming it for prediction...")

            x = self._fill_missing_values(x)
            x = self._cap_employee_length(x)
            x = self._map_default_on_file_column(x)
            x = self._ordinal_encode_columns(x)
            x = self._one_hot_encode_columns(x)
            x = self._drop_id_column(x)
            x = self._convert_boolean_columns(x)

            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            logging.info("Trained model loaded/exists.")
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score
            logging.info(f"F1_Score for this model: {trained_model_f1_score}")

            best_model_f1_score=None
            best_model = self.get_best_model()
            if best_model is not None:
                logging.info(f"Computing F1_Score for production model..")
                y_hat_best_model = best_model.predict(x)
                best_model_f1_score = f1_score(y, y_hat_best_model)
                logging.info(f"F1_Score-Production Model: {best_model_f1_score}, F1_Score-New Trained Model: {trained_model_f1_score}")
            
            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluateModelResponse(trained_model_f1_score=trained_model_f1_score,
                                           best_model_f1_score=best_model_f1_score,
                                           is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                                           difference=trained_model_f1_score - tmp_best_model_score
                                           )
            logging.info(f"Result: {result}")
            return result

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """  
        try:
            print("------------------------------------------------------------------------------------------------")
            logging.info("Initialized Model Evaluation Component.")
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference)

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys) from e