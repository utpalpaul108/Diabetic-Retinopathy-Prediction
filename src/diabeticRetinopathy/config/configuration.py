from diabeticRetinopathy.constants import *
from diabeticRetinopathy.utils import read_yaml, create_directories
from diabeticRetinopathy.entity import DataIngestionConfig, PrepareBaseModelConfig, PrepareCallbacksConfig, TrainingConfig, EvaluationConfig, MLDataIngestionConfig, MLDataPreprocessingConfig, MLModelTrainingConfig
import os

# Configuration Manager
class ConfigurationManager:
    def __init__(self, config_filepath = CONFIG_FILE_PATH, params_filepath = PARAMS_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])
    
    # For ML Model Configuration
    ###############################
    def get_ml_data_ingestion_config(self) -> MLDataIngestionConfig: 
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = MLDataIngestionConfig(
            root_dir = Path(config.root_dir),
            ml_data_source_url = config.ml_data_source_url,
            raw_dataset_dir = Path(config.raw_dataset_dir),
            ml_dataset_dir = Path(config.ml_dataset_dir),
            ml_data_path = Path(config.ml_data_path)
        )

        return data_ingestion_config
    
    def get_ml_data_preprocessing_config(self) -> MLDataPreprocessingConfig:
        config = self.config
        create_directories([config.data_preprocessor.root_dir])

        data_preprocessing_config = MLDataPreprocessingConfig(
            data_path = Path(config.data_ingestion.ml_data_path),
            preprocessor_path = Path(config.data_preprocessor.preprocessor_path)
        )

        return data_preprocessing_config
    
    def get_ml_model_training_config(self) -> MLModelTrainingConfig:
        config= self.config.training
        create_directories([config.root_dir])

        model_training_config = MLModelTrainingConfig(
            best_ml_model_path = Path(config.best_ml_model_path)
        )

        return  model_training_config


    # For DL Model Configuration
    ###############################
    def get_data_ingestion_config(self) -> DataIngestionConfig: 
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir = Path(config.root_dir),
            source_url = config.source_URL,
            raw_dataset_dir = Path(config.raw_dataset_dir),
            dataset_dir = Path(config.dataset_dir)
        )

        return data_ingestion_config
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:

        config = self.config.prepare_base_model
        create_directories([config.root_dir])
        
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir = Path(config.root_dir),
            base_model_path = Path(config.base_model_path),
            updated_base_model_path = Path(config.updated_base_model_path),
            params_image_size = self.params.IMAGE_SIZE,
            params_learning_rate = self.params.LEARNING_RATE,
            params_dropout_rate = self.params.DROPOUT_RATE,
            params_include_top = self.params.INCLUDE_TOP,
            params_weights = self.params.WEIGHTS,
            params_classes = len(self.params.CLASSES)
        )

        return prepare_base_model_config
    
    def get_prepare_callbacks_config(self) ->PrepareCallbacksConfig:
        config = self.config.prepare_callbacks
        model_ckpt_dir = os.path.dirname(config.checkpoint_model_filepath)
        create_directories([Path(config.tensorboard_root_log_dir), Path(model_ckpt_dir)])
        
        prepare_callbacks_config = PrepareCallbacksConfig(
            root_dir = Path(config.root_dir),
            tensorboard_root_log_dir = Path(config.tensorboard_root_log_dir),
            checkpoint_model_filepath = config.checkpoint_model_filepath
        )

        return prepare_callbacks_config
    
    def get_training_config(self) -> TrainingConfig:
        base_model_config = self.config.prepare_base_model
        training_config = self.config.training
        params = self.params
        create_directories([Path(training_config.root_dir)])
        
        training_config = TrainingConfig(
            root_dir = Path(training_config.root_dir),
            trained_model_path = Path(training_config.trained_model_path),
            updated_base_model_path = Path(base_model_config.updated_base_model_path),
            training_path = Path(training_config.training_file_path),
            training_images_path = Path(training_config.training_images_path),
            params_epochs =  params.EPOCHS,
            params_batch_size = params.BATCH_SIZE,
            params_is_augmentation = params.AUGMENTATION,
            params_image_size = params.IMAGE_SIZE
        )

        return training_config
    
    def get_validation_config(self) -> EvaluationConfig:
        
        eval_config = EvaluationConfig(
            path_of_model =  Path(self.config.training.trained_model_path),
            training_path = Path(self.config.training.training_file_path),
            training_images_path = Path(self.config.training.training_images_path),
            all_params = self.params,
            params_image_size = self.params.IMAGE_SIZE,
            params_batch_size = self.params.BATCH_SIZE,
            evaluation_score_path = Path(self.config.evaluation.evaluation_score_path)
        )
        
        return eval_config