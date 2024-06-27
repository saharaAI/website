import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
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
    def __init__(self, path):
        self.path = path
        self.dataset = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        
        self.model = self._build_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def _build_model(self):
        return nn.Sequential(
            nn.Linear(28, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )

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

    def scale_data(self):
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        joblib.dump(self.scaler, './Normalisation_CreditScoring')
        print("Data scaled and scaler saved")

    def _get_sampler(self):
        class_sample_count = np.array([len(np.where(self.y_train == t)[0]) for t in np.unique(self.y_train)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in self.y_train])

        samples_weight = torch.from_numpy(samples_weight)
        samples_weighter = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

        return samples_weighter

    def train_model(self, num_epochs=20, batch_size=64):
        train_dataset = CreditRiskDataset(self.X_train, self.y_train)
        sampler = self._get_sampler()
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        
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
        
        torch.save(self.model.state_dict(), './CreditRiskModel.pth')
        print("Model trained and saved as CreditRiskModel.pth")

    def evaluate_model(self):
        self.model.eval()
        test_dataset = CreditRiskDataset(self.X_test, self.y_test)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        all_preds = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                outputs = self.model(X_batch)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.numpy())
        
        accuracy = accuracy_score(self.y_test, all_preds)
        report = classification_report(self.y_test, all_preds, output_dict=True)
        
        print(confusion_matrix(self.y_test, all_preds))
        print(f"Accuracy: {accuracy}")
        print(f"Classification Report: \n{classification_report(self.y_test, all_preds)}")
        roc = roc_auc_score(self.y_test, all_preds)
        print(f"ROC AUC Score: {roc}")
        return accuracy, report , roc

    def write_output(self):
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
        df_prediction_target = pd.DataFrame(all_preds, columns=['predicted_TARGET'])
        df_test_dataset = pd.DataFrame(self.y_test, columns=['Actual Outcome'])

        dfx = pd.concat([df_test_dataset.reset_index(drop=True), df_prediction_prob, df_prediction_target], axis=1)
        dfx.to_csv("./Model_Prediction.csv", sep=',', encoding='UTF-8')
        print("Output written to Model_Prediction.csv")
        return dfx.head()

# Example usage:
if __name__ == "__main__":
    model = CreditRiskModel('./datasets/Dataset_CreditScoring.xlsx')
    model.load_data()
    model.prepare_data()
    model.split_data(test_size=0.2, random_state=0)
    model.scale_data()
    model.train_model(num_epochs=100, batch_size=64)
    accuracy, report , roc = model.evaluate_model()
    model.write_output()
