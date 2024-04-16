from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    raw_dataset_dir: Path
    dataset_dir: Path


@dataclass(frozen=True)
class DataValidationConfig:
    dataset_dir: Path
    required_file_list: list
    root_dir: Path
    validation_status_file_path: Path


@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    yolo_model_gitgub_url: str
    num_classes: int
    pretrained_model_name: str
    image_size: int
    batch_size: int
    epochs: int
    required_files: list
    dataset_dir: Path



from dataclasses import dataclass
from pathlib import Path

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
    trained_ml_model_path = Path,
    ml_preprocessor_path = Path,
    # trained_model_path: Path
    # updated_base_model_path: Path
    # training_data: Path
    # params_epochs: int
    # params_batch_size: int
    # params_is_augmentation: bool
    # params_image_size: list

@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    all_params: dict
    params_image_size: list
    params_batch_size: int
    evaluation_score_path: Path