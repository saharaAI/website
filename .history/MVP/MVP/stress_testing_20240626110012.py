import pandas as pd
import numpy as np
from scipy.stats import norm 
import streamlit as st

class StressTester:
    def __init__(self):
        pass

    def run_simulation(self, economic_scenario, loan_data):
        """Executes the stress test with a more advanced default probability model."""

        loan_data = loan_data.copy() # Avoid modifying original DataFrame
        # Check if 'interest_rate' column exists in the DataFrame (loan_data)
        if 'interest_rate' not in loan_data.columns:
            loan_data['interest_rate'] = 0.0 
        # 1. Apply Economic Shocks 
        loan_data['stressed_interest_rate'] = loan_data['interest_rate'] + economic_scenario['interest_rate_change']

        # 2. Simulate Default Probabilities (More Advanced Example)
        loan_data['default_probability'] = loan_data.apply(
            lambda row: self.simulate_default_probability(
                row['credit_score'], 
                row['loan_amount'], 
                row['stressed_interest_rate'],
                economic_scenario['unemployment_change']
            ), 
            axis=1 
        )

        # 3. Simulate Loan Defaults
        random_numbers = np.random.rand(len(loan_data))
        loan_data['default'] = np.where(random_numbers < loan_data['default_probability'], 1, 0)

        # 4. Calculate Losses 
        loan_data['loss'] = loan_data['loan_amount'] * loan_data['default'] * loan_data['stressed_interest_rate']

        # 5. Aggregate Results
        total_loss = loan_data['loss'].sum()
        mean_loss_rate = (total_loss / loan_data['loan_amount'].sum()) * 100
        default_rate = (loan_data['default'].sum() / len(loan_data)) * 100

        results = {
            "total_loss": total_loss,
            "mean_loss_rate": mean_loss_rate,
            "default_rate": default_rate
        }
        return results, loan_data # Return the DataFrame with results

    def simulate_default_probability(self, credit_score, loan_amount, interest_rate, unemployment_change):
        """
        A more sophisticated example function to simulate default probability.

        This example considers:
            - Credit Score: Lower scores increase default risk.
            - Loan Amount: Higher amounts increase risk.
            - Interest Rate: Higher rates increase risk.
            - Unemployment Change: Higher unemployment increases risk.

        You should customize this function with your domain knowledge
        and statistical modeling techniques for best results.
        """
        # Define weights for each factor (adjust based on your risk assessment)
        weights = {
            'credit_score': -0.005,  
            'loan_amount': 0.000001, 
            'interest_rate': 0.2,  
            'unemployment': 0.03 
        }

        # Calculate risk score (linear combination, replace with your model)
        risk_score = (
            weights['credit_score'] * credit_score +
            weights['loan_amount'] * loan_amount +
            weights['interest_rate'] * interest_rate +
            weights['unemployment'] * unemployment_change
        )

        # Use a standard normal CDF to convert risk score to probability
        default_probability = norm.cdf(risk_score)  

        # Ensure probability is within 0-1 range
        return max(0, min(default_probability, 1))  

# --- Streamlit App ---
st.title("Stress Testing Module")

# --- Load or simulate your loan data --- 
loan_data = pd.DataFrame({
    'loan_amount': [10000, 25000, 50000, 15000, 30000],
    'interest_rate': [0.05, 0.06, 0.045, 0.07, 0.055],  # Add interest rates
    'credit_score': [720, 680, 750, 650, 700]
})



# --- Collect Stress Test Parameters ---
st.subheader("Economic Scenario Parameters")
interest_rate_change = st.slider("Interest Rate Change (%)", -5.0, 5.0, 0.0, 0.25) / 100
unemployment_change = st.slider("Unemployment Rate Change (%)", -10.0, 10.0, 0.0, 0.5) / 100

economic_scenario = {
    "interest_rate_change": interest_rate_change,
    "unemployment_change": unemployment_change
}

# --- Instantiate and Run Stress Test ---
stress_tester = StressTester()

if st.button("Run Stress Test"):
    results, stressed_loan_data = stress_tester.run_simulation(economic_scenario, loan_data)

    st.subheader("Stress Test Results")
    st.write(f"Total Estimated Loss: ${results['total_loss']:,.2f}")
    st.write(f"Mean Loss Rate: {results['mean_loss_rate']:.2f}%")
    st.write(f"Simulated Default Rate: {results['default_rate']:.2f}%")

    # --- Display the stressed loan data (optional) ---
    st.subheader("Stressed Loan Data")
    st.write(stressed_loan_data)