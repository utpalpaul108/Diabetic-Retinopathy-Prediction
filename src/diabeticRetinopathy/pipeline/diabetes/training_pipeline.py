from diabeticRetinopathy.config import ConfigurationManager
from diabeticRetinopathy.components.diabetes.data_ingestion import DataIngestion
from diabeticRetinopathy.components.diabetes.data_preprocessing import DataPreprocessing
from diabeticRetinopathy.components.diabetes.model_training import ModelTraining


class TrainingPipeline:
    def train(self):
        try:
            config = ConfigurationManager()

            # data_ingestion_config = config.get_ml_data_ingestion_config()
            # data_ingestion = DataIngestion(config=data_ingestion_config)
            # data_ingestion.initiate_data_ingestion()

            data_preprocessing_config = config.get_ml_data_preprocessing_config()
            data_preprocessing = DataPreprocessing(config=data_preprocessing_config)
            X_train, X_test, y_train, y_test = data_preprocessing.initiate_data_preprocessing()

            model_training_config = config.get_ml_model_training_config()
            model_training = ModelTraining(config=model_training_config)
            model_training.initiate_model_training(X_train, X_test, y_train, y_test)

        except Exception as e:
            raise e