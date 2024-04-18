from dataclasses import dataclass
from pathlib import Path

# For ML Model
################################################################
@dataclass(frozen=True)
class MLDataIngestionConfig:
    root_dir: Path
    ml_data_source_url: str
    raw_dataset_dir: Path
    ml_dataset_dir: Path
    ml_data_path: Path

@dataclass(frozen=True)
class MLDataPreprocessingConfig:
    data_path: Path
    preprocessor_path: Path

@dataclass(frozen=True)
class MLModelTrainingConfig:
    best_ml_model_path: Path


# For DL Model
################################################################

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    raw_dataset_dir: Path
    dataset_dir: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_dropout_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int

@dataclass(frozen=True)
class PrepareCallbacksConfig:
    root_dir: Path
    tensorboard_root_log_dir: Path
    checkpoint_model_filepath: Path

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_path: Path
    training_images_path: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list

@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_path: Path
    training_images_path: Path
    all_params: dict
    params_image_size: list
    params_batch_size: int
    evaluation_score_path: Path