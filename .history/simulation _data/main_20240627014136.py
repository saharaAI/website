import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class CreditDataGenerator:
    def __init__(self, num_samples=1000, default_percent=20):
        self.num_samples = num_samples
        self.default_percent = default_percent
        
        # Initialize variables
        self.ID = np.arange(1, num_samples + 1)
        self.x1 = np.random.uniform(18, 70, size=num_samples)  # Age
        self.x2 = np.random.normal(50000, 15000, size=num_samples)  # Yearly income
        self.x3 = np.random.randint(12, 360, size=num_samples)  # Credit length
        self.x4 = np.random.normal(100000, 50000, size=num_samples)  # Amount of loans
        self.x5 = np.random.uniform(0.5, 30, size=num_samples)  # Length of stay
        self.x6 = np.random.choice(['Home', 'Car', 'Education', 'Personal'], size=num_samples,
                                   p=[0.3, 0.2, 0.3, 0.2])  # Purpose
        self.x7 = np.random.choice(['Self-employed', 'Employed', 'Unemployed'], size=num_samples,
                                   p=[0.4, 0.5, 0.1])  # Employment
        self.x8 = np.random.choice(['Own', 'Rent', 'Other'], size=num_samples,
                                   p=[0.6, 0.3, 0.1])  # Type of house
        self.x10 = np.random.choice(['Single', 'Married', 'Divorced'], size=num_samples,
                                    p=[0.4, 0.4, 0.2])  # Marital Status
        self.x11 = np.random.choice(['High school', 'Bachelor', 'Master', 'PhD'], size=num_samples,
                                    p=[0.3, 0.4, 0.2, 0.1])  # Education
        self.x12 = np.random.choice(['0', '1', '2', '3+'], size=num_samples,
                                    p=[0.4, 0.3, 0.2, 0.1])  # Number of dependents
    
    def sigmoid(self, x):
        # Sparse sigmoid to avoid overflow and underflow
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
    
    def ground_truth_pr(self, X, alpha=None):
        if alpha is None:
            alpha = np.random.rand(X.shape[1])  # Generate random alpha if not provided
        
        return self.sigmoid(np.dot(X, alpha))
    
    def simulate_y(self, X, alpha=None):
        pr = self.ground_truth_pr(X, alpha)
        
        # Compute the probability threshold to achieve desired default percentage
        threshold = np.percentile(pr, 100 - self.default_percent)
        
        # Classify y based on the threshold
        y = np.where(pr >= threshold, 1, 0)
        
        return y
    
    def generate_data(self):
        # One-hot encoding categorical variables
        enc = OneHotEncoder(drop='first', sparse=False)
        x6_encoded = enc.fit_transform(self.x6.reshape(-1, 1))
        x7_encoded = enc.fit_transform(self.x7.reshape(-1, 1))
        x8_encoded = enc.fit_transform(self.x8.reshape(-1, 1))
        x10_encoded = enc.fit_transform(self.x10.reshape(-1, 1))
        x11_encoded = enc.fit_transform(self.x11.reshape(-1, 1))
        x12_encoded = enc.fit_transform(self.x12.reshape(-1, 1))
        
        # Combine all variables into X
        X = np.column_stack((self.x1, self.x2, self.x3, self.x4, self.x5,
                             x6_encoded, x7_encoded, x8_encoded,
                             x10_encoded, x11_encoded, x12_encoded))
        
        y = self.simulate_y(X)
        
        # Compute percentages of class 0 and 1
        class_0_percent = np.mean(y == 0) * 100
        class_1_percent = np.mean(y == 1) * 100
        
        # Creating a DataFrame
        data = pd.DataFrame({
            'ID': self.ID,
            'x1': self.x1,
            'x2': self.x2,
            'x3': self.x3,
            'x4': self.x4,
            'x5': self.x5,
            'x6': self.x6,
            'x7': self.x7,
            'x8': self.x8,
            'x10': self.x10,
            'x11': self.x11,
            'x12': self.x12,
            'y': y
        })
        
        return data, class_0_percent, class_1_percent

# Example usage:
if __name__ == "__main__":
    generator = CreditDataGenerator(num_samples=1000, default_percent=30)  # Adjust default_percent as desired
    data, class_0_percent, class_1_percent = generator.generate_data()
    print("Class 0 percentage:", class_0_percent)
    print("Class 1 percentage:", class_1_percent)
    print(data.head())
