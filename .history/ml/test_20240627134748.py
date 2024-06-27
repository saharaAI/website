import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
import joblib
import openpyxl
from imblearn.over_sampling import SMOTE

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
        else:
            raise ValueError("Unsupported model type. Choose 'logistic_regression', 'lightgbm', or 'xgboost'.")

    def load_data(self):
        self.dataset = pd.read_excel(self.path)
        print(f"Data loaded with shape: {self.dataset.shape}")

    def prepare_data(self):
        # Dropping customer ID column from the dataset
        self.dataset = self.dataset.drop('ID', axis=1)
        # Filling missing values with mean
        self.dataset = self.dataset.fillna(self.dataset.mean())
        print(f"Missing values handled. Data shape: {self.dataset.shape}")

    def split_data(self, test_size=0.3, random_state=0):
        y = self.dataset.iloc[:, 0].values
        X = self.dataset.iloc[:, 1:29].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        print("Data split into train and test sets")

    def apply_smote(self):
        smote = SMOTE(random_state=42)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
        print("SMOTE applied to training data")

    def scale_data(self):
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        joblib.dump(self.scaler, './Normalisation_CreditScoring')
        print("Data scaled and scaler saved")

    def train_model(self):
        self.classifier.fit(self.X_train, self.y_train)
        joblib.dump(self.classifier, f'./Classifier_CreditScoring_{self.model_type}')
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
        dfx.to_csv("./Model_Prediction.csv", sep=',', encoding='UTF-8')
        print("Output written to Model_Prediction.csv")
        return dfx.head()

# Function to save benchmark results to an Excel file
def save_benchmark_results(results, output_path='Benchmark_Results.xlsx'):
    df = pd.DataFrame(results)
    df.to_excel(output_path, index=False)
    print(f"Benchmark results saved to {output_path}")

# Example usage:
if __name__ == "__main__":
    model_types = ['logistic_regression', 'lightgbm', 'xgboost']
    benchmark_results = []

    for model_type in model_types:
        print(f"Testing model: {model_type}")
        model = CreditRiskModel('../datasets/Dataset_CreditScoring.xlsx', model_type=model_type)
        model.load_data()
        model.prepare_data()
        model.split_data()
        model.apply_smote()  # Apply SMOTE after splitting the data
        model.scale_data()
        model.train_model()
        accuracy, report = model.evaluate_model()

        result = {
            'Model': model_type,
            'Accuracy': accuracy,
            'Report': report
        }
        benchmark_results.append(result)

    save_benchmark_results(benchmark_results)