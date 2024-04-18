from diabeticRetinopathy.entity import MLModelTrainingConfig
from diabeticRetinopathy.utils import save_object
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import numpy as np


# ModelTraining Component
class ModelTraining:
    def __init__(self, config: MLModelTrainingConfig):
        self.config = config

    def _evaluate_model(self, models, X_train, X_test, y_train, y_test):
        try:
            report = {}
            best_model = {'': -np.inf}

            # Evaluate the models base on the 
            for i in range(len(models)):
                model_name = list(models.keys())[i]
                model = list(models.values())[i]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                if list(best_model.values())[0] < score:
                    best_model = {model_name: score}

                report[model_name] = score

            return report, best_model

        except Exception as e:
            raise e

    def initiate_model_training(self, X_train, X_test, y_train, y_test):
        try:

            # List of the Models
            models = {
                'SVC': SVC(kernel='linear', gamma='scale'),
                'DecisionTree': DecisionTreeClassifier(),
                'RandomForest': RandomForestClassifier(criterion='entropy', max_features='sqrt'),
                'GradientBoosting': GradientBoostingClassifier(criterion='squared_error', loss='exponential'),
                'KNeighbors': KNeighborsClassifier(algorithm='auto',n_neighbors=9, weights='distance')
            }

            print(type(models))

            # Find the best model
            model_report, best_model = self._evaluate_model(models=models, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
            print(model_report)
            print(best_model)
            best_model = models[list(best_model.keys())[0]]

            # Save the best model
            save_object(self.config.best_ml_model_path, best_model)

        except Exception as e:
            raise e
