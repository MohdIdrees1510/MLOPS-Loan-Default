import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

#    def get_data_transformer_object(self) -> Pipeline:
#        """
#        Creates and returns a data transformer object for the data, 
#        including gender mapping, dummy variable creation, column renaming,
#        feature scaling, and type adjustments.
#        """
#        logging.info("Entered get_data_transformer_object method of DataTransformation class")
#
#        try:
#            # Initialize transformers
#            numeric_transformer = StandardScaler()
#            min_max_scaler = MinMaxScaler()
#            logging.info("Transformers Initialized: StandardScaler-MinMaxScaler")
#
#            # Load schema configurations
#            num_features = self._schema_config['num_features']
#            mm_columns = self._schema_config['mm_columns']
#            logging.info("Cols loaded from schema.")
#
#            # Creating preprocessor pipeline
#            preprocessor = ColumnTransformer(
#                transformers=[
#                    ("StandardScaler", numeric_transformer, num_features),
#                    ("MinMaxScaler", min_max_scaler, mm_columns)
#                ],
#                remainder='passthrough'  # Leaves other columns as they are
#            )
#
#            # Wrapping everything in a single pipeline
#            final_pipeline = Pipeline(steps=[("Preprocessor", preprocessor)])
#            logging.info("Final Pipeline Ready!!")
#            logging.info("Exited get_data_transformer_object method of DataTransformation class")
#            return final_pipeline
#
#        except Exception as e:
#            logging.exception("Exception occurred in get_data_transformer_object method of DataTransformation class")
#            raise MyException(e, sys) from e

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


#    def _create_dummy_columns(self, df):
#        """Create dummy variables for categorical features."""
#        logging.info("Creating dummy variables for categorical features")
#        df = pd.get_dummies(df, drop_first=True)
#        return df

#    def _columns_casting(self, df):
#        """Ensure integer types for dummy columns."""
#        logging.info("columns casting to int")
#        df = df.rename(columns={
#            "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
#            "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
#        })
#        for col in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
#            if col in df.columns:
#                df[col] = df[col].astype('int')
#        return df

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


    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation component for the pipeline.
        """
        try:
            logging.info("Data Transformation Started !!!")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # Load train and test data
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train-Test data loaded")

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            logging.info("Input and Target cols defined for both train and test df.")

            # Apply custom transformations in specified sequence
#            input_feature_train_df = self._map_gender_column(input_feature_train_df)
#            input_feature_train_df = self._drop_id_column(input_feature_train_df)
#            input_feature_train_df = self._create_dummy_columns(input_feature_train_df)
#            input_feature_train_df = self._rename_columns(input_feature_train_df)

#            input_feature_test_df = self._map_gender_column(input_feature_test_df)
#            input_feature_test_df = self._drop_id_column(input_feature_test_df)
#            input_feature_test_df = self._create_dummy_columns(input_feature_test_df)
#            input_feature_test_df = self._rename_columns(input_feature_test_df)
#            logging.info("Custom transformations applied to train and test data")


            # TRAIN
            input_feature_train_df = self._fill_missing_values(input_feature_train_df)
            input_feature_train_df = self._cap_employee_length(input_feature_train_df)
            input_feature_train_df = self._map_default_on_file_column(input_feature_train_df)  # or _map_gender_column if reused
            input_feature_train_df = self._ordinal_encode_columns(input_feature_train_df)
            input_feature_train_df = self._one_hot_encode_columns(input_feature_train_df)
            input_feature_train_df = self._drop_id_column(input_feature_train_df)
            input_feature_train_df = self._convert_boolean_columns(input_feature_train_df)
            
            

            # TEST
            input_feature_test_df = self._fill_missing_values(input_feature_test_df)
            input_feature_test_df = self._cap_employee_length(input_feature_test_df)
            input_feature_test_df = self._map_default_on_file_column(input_feature_test_df)  # or _map_gender_column if reused
            input_feature_test_df = self._ordinal_encode_columns(input_feature_test_df)
            input_feature_test_df = self._one_hot_encode_columns(input_feature_test_df)
            input_feature_test_df = self._drop_id_column(input_feature_test_df)
            input_feature_test_df = self._convert_boolean_columns(input_feature_test_df)
            
            
            logging.info("Custom transformations applied to train and test data")

#            logging.info("Starting data transformation")
#            preprocessor = self.get_data_transformer_object()
#            logging.info("Got the preprocessor object")

#            logging.info("Initializing transformation for Training-data")
#            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
#            logging.info("Initializing transformation for Testing-data")
#            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
#            logging.info("Transformation done end to end to train-test df.")

            logging.info("Skipping scaling — using raw transformed features for XGBoost")

            input_feature_train_arr = input_feature_train_df.values
            input_feature_test_arr = input_feature_test_df.values

            logging.info("Converted train and test DataFrames to NumPy arrays")

#            logging.info("Applying SMOTEENN for handling imbalanced dataset.")
#            smt = SMOTEENN(sampling_strategy="minority")
#            input_feature_train_final, target_feature_train_final = smt.fit_resample(
#                input_feature_train_arr, target_feature_train_df
#            )
#            input_feature_test_final, target_feature_test_final = smt.fit_resample(
#                input_feature_test_arr, target_feature_test_df
#            )
#            logging.info("SMOTEENN applied to train-test df.")

#            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
#            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]
#            logging.info("feature-target concatenation done for train-test df."

#            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
#            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
#            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
#            logging.info("Saving transformation object and transformed files.")

#            logging.info("Data transformation completed successfully")
#            return DataTransformationArtifact(
#                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
#                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
#                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
#            )

            logging.info("Applying SMOTEENN for handling imbalanced dataset.")
            smt = SMOTEENN(sampling_strategy="minority")
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
            input_feature_train_arr, target_feature_train_df
            )
            logging.info("SMOTEENN applied to training data")

            # Keep test data untouched
            input_feature_test_final = input_feature_test_arr
            target_feature_test_final = target_feature_test_df

            # Concatenate features and targets
            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]
            logging.info("Feature-target concatenation done for train-test df.")

            # Save transformed arrays
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            logging.info("Saved transformed train and test arrays")

            # Skip saving preprocessor if unused
            # save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)

            logging.info("Data transformation completed successfully")
            return DataTransformationArtifact(
            transformed_object_file_path=None,  # or remove this field if not needed
            transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
            transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise MyException(e, sys) from e