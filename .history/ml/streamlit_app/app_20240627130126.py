# Import libraries
import streamlit as st
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
import base64  # For downloading files
import seaborn as sns
import matplotlib.pyplot as plt

# Hide Streamlit footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """

st.set_page_config(layout="wide")

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Function to get download link for a file
def get_table_download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download {filename} Data</a>'
    return href

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
            raise ValueError(
                "Unsupported model type. Choose 'logistic_regression', 'lightgbm', or 'xgboost'."
            )

    def load_data(self):
        self.dataset = pd.read_excel(self.path)
        print(f"Data loaded with shape: {self.dataset.shape}")

    def prepare_data(self):
        self.dataset = self.dataset.drop('ID', axis=1)
        self.dataset = self.dataset.fillna(self.dataset.mean())
        print(f"Missing values handled. Data shape: {self.dataset.shape}")

    def split_data(self, test_size=0.3, random_state=0):
        y = self.dataset.iloc[:, 0].values
        X = self.dataset.iloc[:, 1:29].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print("Data split into train and test sets")

    def scale_data(self):
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        joblib.dump(self.scaler, './Normalisation_CreditScoring')
        print("Data scaled and scaler saved")

    def train_model(self):
        self.classifier.fit(self.X_train, self.y_train)
        joblib.dump(
            self.classifier, f'./Classifier_CreditScoring_{self.model_type}'
        )
        print(f"Model trained and saved as Classifier_CreditScoring_{self.model_type}")

    def evaluate_model(self):
        y_pred = self.classifier.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred, output_dict=True)
        print(confusion_matrix(self.y_test, y_pred))
        print(f"Accuracy: {accuracy}")
        print(
            f"Classification Report: \n{classification_report(self.y_test, y_pred)}"
        )
        return accuracy, report

    def write_output(self):
        predictions = self.classifier.predict_proba(self.X_test)
        df_prediction_prob = pd.DataFrame(predictions, columns=['prob_0', 'prob_1'])
        df_prediction_target = pd.DataFrame(
            self.classifier.predict(self.X_test), columns=['predicted_TARGET']
        )
        df_test_dataset = pd.DataFrame(self.y_test, columns=['Actual Outcome'])

        dfx = pd.concat(
            [df_test_dataset, df_prediction_prob, df_prediction_target], axis=1
        )
        dfx.to_csv("./Model_Prediction.csv", sep=',', encoding='UTF-8')
        print("Output written to Model_Prediction.csv")
        return dfx.head()


# --- Streamlit App ---

st.title("Credit Risk Scoring Dashboard")
# wide layout

# Model Selection
model_type = st.sidebar.selectbox(
    "Select Model:", ["logistic_regression", "lightgbm", "xgboost"], index=0
)

# File Uploader
uploaded_file = st.sidebar.file_uploader("Choose an Excel file", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Load data (assuming first sheet)
        data = pd.read_excel(uploaded_file)

        # Display data
        st.subheader("Uploaded Data")
        st.write(data.head())

        # Display summary statistics
        st.subheader("Summary Statistics")
        st.write(data.describe())

        # Create and train the model
        model = CreditRiskModel(
            "", model_type=model_type
        )  # Initialize with empty path
        model.dataset = data  # Assign the uploaded data
        model.prepare_data()
        model.split_data()
        model.scale_data()
        model.train_model()

        ## Display results in columns
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Model Evaluation")
            accuracy, report = model.evaluate_model()
            st.write(f"Accuracy: {accuracy:.2f}")
            st.write("Classification Report:")
            st.text(
                classification_report(model.y_test, model.classifier.predict(model.X_test))
            )

            # Confusion Matrix
            st.subheader("Confusion Matrix")
            conf_matrix = confusion_matrix(model.y_test, model.classifier.predict(model.X_test))
            st.write(conf_matrix)

        with col2:
            # Predictions
            st.subheader("Predictions")
            predictions_df = model.write_output()
            st.write(predictions_df)

            # Add download link for predictions
            st.markdown(
                get_table_download_link(predictions_df, "Model_Predictions"),
                unsafe_allow_html=True,
            )
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# --- Benchmarking ---
if st.sidebar.checkbox("Run Benchmark ..."):
    st.subheader("Benchmark Results")

    benchmark_results = []
    model_types = ['logistic_regression', 'lightgbm', 'xgboost']

    for model_type in model_types:
        st.write(f"**Testing model:** {model_type}")  # Indicate which model is running

        model = CreditRiskModel("", model_type=model_type)
        model.dataset = data  # Use the same uploaded data for each model
        model.prepare_data()
        model.split_data()
        model.scale_data()
        model.train_model()
        accuracy, report = model.evaluate_model()

        result = {
            'Model': model_type,
            'Accuracy': accuracy,
            'Precision (Class 1)': report['1']['precision'],
            'Recall (Class 1)': report['1']['recall'],
            'F1-Score (Class 1)': report['1']['f1-score']
        }
        benchmark_results.append(result)

    # Display benchmark results in a table
    benchmark_df = pd.DataFrame(benchmark_results)
    st.table(benchmark_df)
