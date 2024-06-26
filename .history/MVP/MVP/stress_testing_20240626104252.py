import pandas as pd
import numpy as np
from scipy.stats import norm  # For simulating default probabilities

class StressTester:
    def __init__(self):
        pass

    def run_simulation(self, economic_scenario, loan_data):
        """
        Executes the stress test simulation.

        Args:
            economic_scenario (dict): Defines the economic shocks:
                - 'interest_rate_change':  Percentage point change in interest rate (e.g., 0.02 for +2%).
                - 'unemployment_change':  Percentage point change in unemployment rate.
            loan_data (pd.DataFrame): Loan portfolio data with columns:
                - 'loan_amount': The original loan amount.
                - 'interest_rate':  The current interest rate on the loan.
                - 'credit_score': Borrower's credit score (used for default probability). 

        Returns:
            dict: Stress test results, including:
                - 'total_loss': Estimated total portfolio loss.
                - 'mean_loss_rate':  Average loss rate across loans.
                - 'default_rate':  The simulated default rate. 
        """ 

        # --- 1. Apply Economic Shocks ---
        loan_data['stressed_interest_rate'] = loan_data['interest_rate'] + economic_scenario['interest_rate_change']

        # --- 2. Simulate Default Probabilities (Example) ---
        # You'll likely replace this with a more sophisticated model 
        # based on your data and risk assessment. 
        loan_data['default_probability'] = loan_data['credit_score'].apply(
            lambda score: self.simulate_default_probability(score, economic_scenario['unemployment_change'])
        )

        # --- 3. Simulate Loan Defaults --- 
        random_numbers = np.random.rand(len(loan_data))
        loan_data['default'] = np.where(random_numbers < loan_data['default_probability'], 1, 0)

        # --- 4. Calculate Losses ---
        loan_data['loss'] = loan_data['loan_amount'] * loan_data['default'] * loan_data['stressed_interest_rate'] 

        # --- 5. Aggregate Results ---
        total_loss = loan_data['loss'].sum()
        mean_loss_rate = (total_loss / loan_data['loan_amount'].sum()) * 100  # Percentage
        default_rate = (loan_data['default'].sum() / len(loan_data)) * 100 # Percentage

        results = {
            "total_loss": total_loss,
            "mean_loss_rate": mean_loss_rate,
            "default_rate": default_rate
        }
        return results

    def simulate_default_probability(self, credit_score, unemployment_change):
        """
        A simple example function to simulate default probability 
        based on credit score and unemployment change.

        Replace this with your actual default probability model!
        """
        # --- Example: Assume lower scores and higher unemployment increase default risk ---
        base_default_prob = 0.01  # Base default probability
        score_sensitivity = 0.0005 # How much default prob. changes with 1 point of credit score
        unemployment_sensitivity = 0.02  # How much default prob. changes with 1% unemployment change 

        default_prob = base_default_prob - (credit_score * score_sensitivity) + (unemployment_change * unemployment_sensitivity)
        return max(0, min(default_prob, 1))  # Ensure probability is between 0 and 1