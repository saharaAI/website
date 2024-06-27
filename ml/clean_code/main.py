import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
import xgboost as xgb
import joblib
import openpyxl
from sklearn.impute import SimpleImputer  # For handling missing values
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE

class CreditRiskModel:
    def __init__(self, path, model_type='logistic_regression'):
        self.path = path
        self.dataset = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.model_type = model_type

        if model_type == 'logistic_regression':
            self.classifier = LogisticRegression()
        elif model_type == 'lightgbm':
            self.classifier = lgb.LGBMClassifier()
        elif model_type == 'xgboost':
            self.classifier = xgb.XGBClassifier()
        elif model_type == 'random_forest':
            self.classifier = RandomForestClassifier()
        elif model_type == 'decision_tree':
            self.classifier = DecisionTreeClassifier()
        else:
            raise ValueError("Unsupported model type. Choose 'logistic_regression', 'lightgbm', 'xgboost', 'random_forest', or 'decision_tree'.")

    def load_data(self):
        self.dataset = pd.read_excel(self.path)
        print(f"Data loaded with shape: {self.dataset.shape}")

    def prepare_data(self, strategy='mean'):
        # Dropping customer ID column from the dataset
        self.dataset = self.dataset.drop('ID', axis=1)
        
        # Handling missing values
        if strategy == 'mean':
            self.dataset = self.dataset.fillna(self.dataset.mean())
        elif strategy == 'median':
            self.dataset = self.dataset.fillna(self.dataset.median())
        elif strategy == 'mode':
            self.dataset = self.dataset.fillna(self.dataset.mode().iloc[0])  # Replace with mode
        
        print(f"Missing values handled using {strategy}. Data shape: {self.dataset.shape}")

    def split_data(self, test_size=0.3, random_state=0):
        y = self.dataset.iloc[:, 0].values
        X = self.dataset.iloc[:, 1:29].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        print("Data split into train and test sets")

    def apply_smote(self):
        smote = SMOTE(random_state=0)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
        print("Applied SMOTE to the training data")

    def apply_adasyn(self):
        adasyn = ADASYN(random_state=0)
        self.X_train, self.y_train = adasyn.fit_resample(self.X_train, self.y_train)
        print("Applied ADASYN to the training data")

    def apply_borderline_smote(self):
        borderline_smote = BorderlineSMOTE(random_state=0)
        self.X_train, self.y_train = borderline_smote.fit_resample(self.X_train, self.y_train)
        print("Applied BorderlineSMOTE to the training data")

    def scale_data(self):
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        print("Data scaled and scaler saved")

    def hyperparameter_tuning(self):
        if self.model_type == 'random_forest':
            param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}
            grid_search = GridSearchCV(self.classifier, param_grid, cv=5)
            grid_search.fit(self.X_train, self.y_train)
            self.classifier = grid_search.best_estimator_
        elif self.model_type == 'xgboost':
            param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 6, 9]}
            grid_search = GridSearchCV(self.classifier, param_grid, cv=5)
            grid_search.fit(self.X_train, self.y_train)
            self.classifier = grid_search.best_estimator_
        elif self.model_type == 'lightgbm':
            param_grid = {'n_estimators': [50, 100, 200], 'num_leaves': [31, 64, 128]}
            grid_search = GridSearchCV(self.classifier, param_grid, cv=5)
            grid_search.fit(self.X_train, self.y_train)
            self.classifier = grid_search.best_estimator_
        elif self.model_type == 'logistic_regression':
            param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
            grid_search = GridSearchCV(self.classifier, param_grid, cv=5)
            grid_search.fit(self.X_train, self.y_train)
            self.classifier = grid_search.best_estimator_
        elif self.model_type == 'decision_tree':
            param_grid = {'max_depth': [None, 10, 20, 30]}
            grid_search = GridSearchCV(self.classifier, param_grid, cv=5)
            grid_search.fit(self.X_train, self.y_train)
            self.classifier = grid_search.best_estimator_
        print(f"Hyperparameters tuned for {self.model_type}")

    def train_model(self):
        self.classifier.fit(self.X_train, self.y_train)
        print(f"Model trained and saved as Classifier_CreditScoring_{self.model_type}")

    def evaluate_model(self):
        y_pred = self.classifier.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred, output_dict=True)
        print(confusion_matrix(self.y_test, y_pred))
        print(f"Accuracy: {accuracy}")
        print(f"Classification Report: \n{classification_report(self.y_test, y_pred)}")
        return accuracy, report

    def write_output(self):
        predictions = self.classifier.predict_proba(self.X_test)
        df_prediction_prob = pd.DataFrame(predictions, columns=['prob_0', 'prob_1'])
        df_prediction_target = pd.DataFrame(self.classifier.predict(self.X_test), columns=['predicted_TARGET'])
        df_test_dataset = pd.DataFrame(self.y_test, columns=['Actual Outcome'])

        dfx = pd.concat([df_test_dataset, df_prediction_prob, df_prediction_target], axis=1)
        dfx.to_csv(os.path.join('.', 'Model_Prediction.csv'), sep=',', encoding='UTF-8')
        print("Output written to Model_Prediction.csv")
        return dfx.head()

# Function to save benchmark results to a JSON file
def save_benchmark_results(results, output_path='Benchmark_Results.json'):
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Benchmark results saved to {output_path}")

# Example usage:
if __name__ == "__main__":
    model_types = ['logistic_regression', 'lightgbm', 'xgboost', 'random_forest', 'decision_tree']
    resampling_techniques = ['smote', 'adasyn', 'borderline_smote']
    missing_strategy = ['mean', 'median', 'mode']
    benchmark_results = []

    # Get the current working directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the absolute path to the dataset
    dataset_path = os.path.join(current_dir, '..', 'datasets', 'Dataset_CreditScoring.xlsx')

    for strategy in missing_strategy:
        print(f"Handling missing values using strategy: {strategy}")
        for resampling_technique in resampling_techniques:
            print(f"Using resampling technique: {resampling_technique}")
            for model_type in model_types:
                print(f"Testing model: {model_type} with {resampling_technique} and {strategy}")
                model = CreditRiskModel(dataset_path, model_type=model_type)
                model.load_data()
                model.prepare_data(strategy=strategy)
                model.split_data()
                
                if resampling_technique == 'smote':
                    model.apply_smote()
                elif resampling_technique == 'adasyn':
                    model.apply_adasyn()
                elif resampling_technique == 'borderline_smote':
                    model.apply_borderline_smote()
                
                model.scale_data()
                model.hyperparameter_tuning()
                model.train_model()
                accuracy, report = model.evaluate_model()

                result = {
                    'Model': model_type,
                    'Resampling Technique': resampling_technique,
                    'Missing Strategy': strategy,
                    'Accuracy': accuracy,
                    'Report': report
                }
                benchmark_results.append(result)

    save_benchmark_results(benchmark_results)
