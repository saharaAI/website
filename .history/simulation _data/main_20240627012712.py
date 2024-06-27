import pandas as pd
import numpy as np

class CreditDataGenerator:
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        
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
        return np.exp(x) / (1 + np.exp(x))
    
    def ground_truth_pr(self, X, alpha=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 0.9]):
        X_array = np.array(X)
        alpha_array = np.array(alpha)
        return self.sigmoid(np.dot(X_array, alpha_array))
    
    def simulate_y(self, X, alpha=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 0.9]):
        pr = self.ground_truth_pr(X, alpha)
        return np.random.binomial(1, pr)
    
    def generate_data(self):
        X = np.column_stack((self.x1, self.x2, self.x3, self.x4, self.x5,
                             self.x6, self.x7, self.x8, self.x10, self.x11, self.x12))
        y = self.simulate_y(X)
        
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
        
        return data

# Example usage:
if __name__ == "__main__":
    generator = CreditDataGenerator(num_samples=1000)
    data = generator.generate_data()
    print(data.head())