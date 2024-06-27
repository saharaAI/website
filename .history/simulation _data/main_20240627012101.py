import pandas as pd
import numpy as np

# Define the number of rows for your dataset
num_samples = 1000

# Simulate data for each variable with ground truth distributions

# ID variable (sequential ID)
ID = np.arange(1, num_samples + 1)

# x1: Age in years plus twelfths of a year (Numerical)
x1 = np.random.uniform(18, 70, size=num_samples)  # Example range for age

# x2: Yearly income (in Dinars) (Numerical)
x2 = np.random.normal(50000, 15000, size=num_samples)  # Example mean income 50,000 Dinars with SD 15,000

# x3: Credit length (in months) (Numerical)
x3 = np.random.randint(12, 360, size=num_samples)  # Example range for credit length in months

# x4: Amount of loans (in Dinars) (Numerical)
x4 = np.random.normal(100000, 50000, size=num_samples)  # Example mean loan amount 100,000 Dinars with SD 50,000

# x5: Length of stay (in years) (Numerical)
x5 = np.random.uniform(0.5, 30, size=num_samples)  # Example range for length of stay in years

# x6: Purpose (Categorical)
purposes = ['Home', 'Car', 'Education', 'Personal']
x6 = np.random.choice(purposes, size=num_samples, p=[0.3, 0.2, 0.3, 0.2])  # Adjust probabilities based on distribution

# x7: Employment (Categorical)
employment_types = ['Self-employed', 'Employed', 'Unemployed']
x7 = np.random.choice(employment_types, size=num_samples, p=[0.4, 0.5, 0.1])  # Adjust probabilities based on distribution

# x8: Type of house (Categorical)
house_types = ['Own', 'Rent', 'Other']
x8 = np.random.choice(house_types, size=num_samples, p=[0.6, 0.3, 0.1])  # Adjust probabilities based on distribution

# x10: Marital Status (Categorical)
marital_statuses = ['Single', 'Married', 'Divorced']
x10 = np.random.choice(marital_statuses, size=num_samples, p=[0.4, 0.4, 0.2])  # Adjust probabilities based on distribution

# x11: Education (Categorical)
education_levels = ['High school', 'Bachelor', 'Master', 'PhD']
x11 = np.random.choice(education_levels, size=num_samples, p=[0.3, 0.4, 0.2, 0.1])  # Adjust probabilities based on distribution

# x12: Number of dependents (Categorical)
dependents = ['0', '1', '2', '3+']
x12 = np.random.choice(dependents, size=num_samples, p=[0.4, 0.3, 0.2, 0.1])  # Adjust probabilities based on distribution

# Simulating y based on a custom distribution function
def simulate_y(x1, x2, x3, x4, x5, x6, x7, x8, x10, x11, x12):
    ## Define your custom distribution function here
    pass
# Generate values for y based on the custom function
y = simulate_y(x1, x2, x3, x4, x5, x6, x7, x8, x10, x11, x12)

# Creating a DataFrame
data = pd.DataFrame({
    'ID': ID,
    'x1': x1,
    'x2': x2,
    'x3': x3,
    'x4': x4,
    'x5': x5,
    'x6': x6,
    'x7': x7,
    'x8': x8,
    'x10': x10,
    'x11': x11,
    'x12': x12,
    'y': y
})

# Displaying the first few rows of the generated data
print(data.head())
