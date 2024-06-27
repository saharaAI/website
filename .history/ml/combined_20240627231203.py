import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
import joblib

class CreditRiskDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CreditRiskModel:
    def __init__(self, path, model_type='neural_network'):
        self.path = path
        self.dataset = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.model_type = model_type

        if model_type == 'neural_network':
            self.model = self._build_neural_network()
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression()
        elif model_type == 'lightgbm':
            self.model = lgb.LGBMClassifier()
        elif model_type == 'xgboost':
            self.model = xgb.XGBClassifier()
        else:
            raise ValueError("Unsupported model type. Choose 'neural_network', 'logistic_regression', 'lightgbm', or 'xgboost'.")

    def _build_neural_network(self, hidden_layers=[64, 32, 32]):
        layers = []
        input_size = 28  # Assuming 28 features after preprocessing
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 2))  # Binary classification
        return nn.Sequential(*layers)

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

     def _get_sampler(self, data_len):  # Add data_len as argument
        class_sample_count = np.array([len(np.where(self.y_train == t)[0]) for t in np.unique(self.y_train)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in self.y_train])

        samples_weight = torch.from_numpy(samples_weight)
        samples_weighter = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), data_len)  # Use data_len

        return samples_weighter

    def _train_neural_network(self, num_epochs, batch_size, k_folds):
        kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.X_train, self.y_train)):
            print(f"Fold {fold + 1}/{k_folds}")

            X_train_fold, y_train_fold = self.X_train[train_idx], self.y_train[train_idx]
            X_val_fold, y_val_fold = self.X_train[val_idx], self.y_train[val_idx]

            train_dataset = CreditRiskDataset(X_train_fold, y_train_fold)
            
            # Create sampler here, inside the loop
            sampler = self._get_sampler(len(train_dataset)) # Pass the length of the fold data

            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    def train_model(self, num_epochs=20, batch_size=64, k_folds=5):
        if self.model_type == 'neural_network':
            self._train_neural_network(num_epochs, batch_size, k_folds)
        else:
            self._train_traditional_model(k_folds)

    def _train_neural_network(self, num_epochs, batch_size, k_folds):
        kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.X_train, self.y_train)):
            print(f"Fold {fold + 1}/{k_folds}")

            X_train_fold, y_train_fold = self.X_train[train_idx], self.y_train[train_idx]
            X_val_fold, y_val_fold = self.X_train[val_idx], self.y_train[val_idx]

            train_dataset = CreditRiskDataset(X_train_fold, y_train_fold)
            sampler = self._get_sampler()
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
            
            val_dataset = CreditRiskDataset(X_val_fold, y_val_fold)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            self.model.train()
            for epoch in range(num_epochs):
                epoch_loss = 0
                for X_batch, y_batch in train_loader:
                    self.optimizer.zero_grad()
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader)}")

                # Validation loop
                self.model.eval()
                val_preds = []
                with torch.no_grad():
                    for X_batch, _ in val_loader:
                        outputs = self.model(X_batch)
                        _, preds = torch.max(outputs, 1)
                        val_preds.extend(preds.numpy())
                val_accuracy = accuracy_score(y_val_fold, val_preds)
                print(f"Validation Accuracy: {val_accuracy}")

            torch.save(self.model.state_dict(), f'./CreditRiskModel_{self.model_type}_fold_{fold + 1}.pth')
            print(f"Model for fold {fold + 1} saved as CreditRiskModel_{self.model_type}_fold_{fold + 1}.pth")
        
    def _train_traditional_model(self, k_folds=5):
        kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.X_train, self.y_train)):
            print(f"Fold {fold + 1}/{k_folds}")

            X_train_fold, y_train_fold = self.X_train[train_idx], self.y_train[train_idx]
            X_val_fold, y_val_fold = self.X_train[val_idx], self.y_train[val_idx]
            
            self.model.fit(X_train_fold, y_train_fold)

            # Validation loop
            val_preds = self.model.predict(X_val_fold)
            val_accuracy = accuracy_score(y_val_fold, val_preds)
            print(f"Validation Accuracy: {val_accuracy}")

            joblib.dump(self.model, f'./Classifier_CreditScoring_{self.model_type}_fold_{fold + 1}')
            print(f"Model for fold {fold + 1} saved as Classifier_CreditScoring_{self.model_type}_fold_{fold + 1}")

    def evaluate_model(self):
        if self.model_type == 'neural_network':
            self.model.eval()
            test_dataset = CreditRiskDataset(self.X_test, self.y_test)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            
            all_preds = []
            with torch.no_grad():
                for X_batch, _ in test_loader:
                    outputs = self.model(X_batch)
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.numpy())
            
            y_pred = all_preds
        else:
            y_pred = self.model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred, output_dict=True)
        print(confusion_matrix(self.y_test, y_pred))
        print(f"Accuracy: {accuracy}")
        print(f"Classification Report: \n{classification_report(self.y_test, y_pred)}")

        roc = roc_auc_score(self.y_test, y_pred)
        print(f"ROC AUC Score: {roc}")
        return accuracy, report, roc
    
    def predict(self, X):
        if self.model_type == 'neural_network':
            self.model.eval()
            X = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                outputs = self.model(X)
                _, preds = torch.max(outputs, 1)
            return preds.numpy()
        else:
            return self.model.predict(X)

    def write_output(self):
        if self.model_type == 'neural_network':
            self.model.eval()
            test_dataset = CreditRiskDataset(self.X_test, self.y_test)
            test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
            
            all_preds = []
            all_probs = []
            with torch.no_grad():
                for X_batch, _ in test_loader:
                    outputs = self.model(X_batch)
                    probs = torch.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.numpy())
                    all_probs.extend(probs.numpy())
            
            df_prediction_prob = pd.DataFrame(all_probs, columns=['prob_0', 'prob_1'])
        else:
            predictions = self.model.predict_proba(self.X_test)
            df_prediction_prob = pd.DataFrame(predictions, columns=['prob_0', 'prob_1'])

        df_prediction_target = pd.DataFrame(self.predict(self.X_test), columns=['predicted_TARGET'])
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
    model_types = ['neural_network', 'logistic_regression', 'lightgbm', 'xgboost']
    benchmark_results = []

    for model_type in model_types:
        print(f"Testing model: {model_type}")
        model = CreditRiskModel('./datasets/Dataset_CreditScoring.xlsx', model_type=model_type)
        model.load_data()
        model.prepare_data()
        model.split_data()
        model.scale_data()
        model.train_model(num_epochs=100, batch_size=64, k_folds=5)  # Adjust as needed
        accuracy, report, roc = model.evaluate_model()
        result = {
            'Model': model_type,
            'Accuracy': accuracy,
            'ROC AUC': roc,
            'Report': report
        }
        
        benchmark_results.append(result)
        model.write_output()

    save_benchmark_results(benchmark_results)