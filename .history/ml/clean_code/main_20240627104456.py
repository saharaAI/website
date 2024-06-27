import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
import joblib

class CreditRiskModel:
    def __init__(self, path):
        self.path = path
        self.dataset = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression()

    def load_data(self):
        self.dataset = pd.read_excel(self.path)
        print(f"Data loaded with shape: {self.dataset.shape}")

    def prepare_data(self):
        # Dropping customer ID column from the dataset
        self.dataset = self.dataset.drop('ID', axis=1)
        # Filling missing values with mean
        self.dataset = self.dataset.fillna(self.dataset.mean())
        print(f"Missing values handled. Data shape: {self.dataset.shape}")

    def split_data(self, test_size=0.2, random_state=0):
        y = self.dataset.iloc[:, 0].values
        X = self.dataset.iloc[:, 1:29].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        print("Data split into train and test sets")

    def scale_data(self):
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        joblib.dump(self.scaler, './Normalisation_CreditScoring')
        print("Data scaled and scaler saved")

    def train_model(self):
        self.classifier.fit(self.X_train, self.y_train)
        joblib.dump(self.classifier, './Classifier_CreditScoring')
        print("Model trained and saved")

    def evaluate_model(self):
        y_pred = self.classifier.predict(self.X_test)
        print(confusion_matrix(self.y_test, y_pred))
        print(f"Accuracy: {accuracy_score(self.y_test, y_pred)}")

    def write_output(self):
        predictions = self.classifier.predict_proba(self.X_test)
        df_prediction_prob = pd.DataFrame(predictions, columns=['prob_0', 'prob_1'])
        df_prediction_target = pd.DataFrame(self.classifier.predict(self.X_test), columns=['predicted_TARGET'])
        df_test_dataset = pd.DataFrame(self.y_test, columns=['Actual Outcome'])

        dfx = pd.concat([df_test_dataset, df_prediction_prob, df_prediction_target], axis=1)
        dfx.to_csv("./Model_Prediction.csv", sep=',', encoding='UTF-8')
        print("Output written to Model_Prediction.csv")
        return dfx.head()

# Example usage:
if __name__ == "__main__":
    model = CreditRiskModel('../datasets/Dataset_CreditScoring.xlsx')
    model.load_data()
    model.prepare_data()
    model.split_data()
    model.scale_data()
    model.train_model()
    model.evaluate_model()
    model.write_output()
