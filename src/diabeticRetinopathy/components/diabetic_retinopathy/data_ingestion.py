import opendatasets as od
import shutil
import os
from PIL import Image
from diabeticRetinopathy.utils import create_directories
from diabeticRetinopathy.entity import DataIngestionConfig

# Data Ingestion Component
class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    
    def _download_dataset(self):
        if not os.path.exists(self.config.dataset_dir) or (not os.listdir(self.config.dataset_dir)):
            try:
                create_directories([self.config.raw_dataset_dir])
                # Download the dataset
                od.download(self.config.source_url,data_dir=self.config.raw_dataset_dir)
            except Exception as e:
                raise e
        

    def _preprocess_dataset(self):
        if os.path.exists(self.config.raw_dataset_dir):
            try:
                create_directories([self.config.dataset_dir])
                for root, dirs, files in os.walk(self.config.raw_dataset_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        
                        # Check if the file is an image and valid
                        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.csv')):
                            # Extract the folder name
                            folder_name = os.path.basename(root)
                            
                            # Create the destination folder if it doesn't exist
                            destination_folder = os.path.join(self.config.dataset_dir, folder_name)
                            create_directories([destination_folder])
                            
                            # Copy the valid image to the destination folder
                            shutil.copy(file_path, os.path.join(destination_folder, file))

                shutil.rmtree(self.config.raw_dataset_dir)
            
            except Exception as e:
                raise e
            
    
    def initiate_data_ingestion(self):
        self._download_dataset()
        self._preprocess_dataset()

