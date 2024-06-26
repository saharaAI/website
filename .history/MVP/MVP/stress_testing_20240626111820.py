import pandas as pd
import numpy as np
from scipy.stats import norm
import streamlit as st
import plotly.graph_objects as go 
class StressTester:
    def __init__(self):
        pass
    def run_simulation(self, economic_scenarios, loan_data):
        """
        Executes stress tests for multiple economic scenarios.

        Args:
            economic_scenarios (list of dict):  List of scenarios, each 
                a dictionary defining economic shocks:
                    - 'name': Scenario name (for labels)
                    - 'interest_rate_change': Change in interest rate (e.g., 0.02 for +2%).
                    - 'unemployment_change': Change in unemployment rate.
                    - 'gdp_growth_change': Change in GDP growth rate.
                    - 'inflation_change': Change in inflation rate. 
            loan_data (pd.DataFrame): Loan data (same as before).

        Returns:
            pd.DataFrame: DataFrame with original loan data and 
                          stress test results for each scenario.
        """
        results = []

        loan_data = loan_data.copy()
        if 'interest_rate' not in loan_data.columns:
            loan_data['interest_rate'] = 0.0

        for scenario in economic_scenarios:
            scenario_results = loan_data.copy()
            scenario_results['stressed_interest_rate'] = loan_data['interest_rate'] + scenario['interest_rate_change']

            # Apply scenario-specific changes (You might need to adjust these calculations)
            scenario_results['default_probability'] = scenario_results.apply(
                lambda row: self.simulate_default_probability(
                    row['credit_score'],
                    row['loan_amount'],
                    row['stressed_interest_rate'],
                    scenario['unemployment_change'] + scenario.get('gdp_growth_change', 0.0) + scenario.get('inflation_change', 0.0)
                ),
                axis=1
            )
            random_numbers = np.random.rand(len(scenario_results))
            scenario_results[f'default_{scenario["name"]}'] = np.where(random_numbers < scenario_results['default_probability'], 1, 0)
            scenario_results[f'loss_{scenario["name"]}'] = scenario_results['loan_amount'] * scenario_results[f'default_{scenario["name"]}'] * scenario_results['stressed_interest_rate']
            results.append(scenario_results)

        # Combine results into a single DataFrame
        combined_results = pd.concat(results, axis=1)
        return combined_results

# --- Streamlit App ---
st.title("Stress Testing Module")

# --- Load or simulate your loan data --- 
loan_data = pd.DataFrame({
    'loan_amount': [10000, 25000, 50000, 15000, 30000],
    'interest_rate': [0.05, 0.06, 0.045, 0.07, 0.055],
    'credit_score': [720, 680, 750, 650, 700]
})

# --- Collect Stress Test Parameters ---
st.subheader("Economic Scenario Parameters")
st.markdown("Define your base economic scenario:")
base_interest_rate_change = st.slider("Interest Rate Change (%)", -5.0, 5.0, 0.0, 0.25) / 100
base_unemployment_change = st.slider("Unemployment Rate Change (%)", -10.0, 10.0, 0.0, 0.5) / 100
base_gdp_growth_change = st.slider("GDP Growth Change (%)", -5.0, 5.0, 0.0, 0.25) / 100
base_inflation_change = st.slider("Inflation Change (%)", -5.0, 5.0, 0.0, 0.25) / 100

# Create a list to store the scenarios
scenarios = []
num_scenarios = st.number_input("Number of scenarios", min_value=1, max_value=5, value=2)
for i in range(num_scenarios):
    st.markdown(f"**Scenario {i+1} Adjustments:**")
    interest_rate_adjust = st.slider(f"Interest Rate Adjustment (Scenario {i+1}) (%)", -5.0, 5.0, 0.0, 0.25) / 100
    unemployment_adjust = st.slider(f"Unemployment Rate Adjustment (Scenario {i+1}) (%)", -10.0, 10.0, 0.0, 0.5) / 100
    gdp_growth_adjust = st.slider(f"GDP Growth Adjustment (Scenario {i+1}) (%)", -5.0, 5.0, 0.0, 0.25) / 100
    inflation_adjust = st.slider(f"Inflation Adjustment (Scenario {i+1}) (%)", -5.0, 5.0, 0.0, 0.25) / 100

    # Append the scenario parameters to the list
    scenarios.append({
        'name': f'Scenario {i+1}',
        'interest_rate_change': base_interest_rate_change + interest_rate_adjust,
        'unemployment_change': base_unemployment_change + unemployment_adjust,
        'gdp_growth_change': base_gdp_growth_change + gdp_growth_adjust,
        'inflation_change': base_inflation_change + inflation_adjust
    })

# --- Instantiate and Run Stress Test ---
stress_tester = StressTester()

if st.button("Run Stress Test"):
    results = stress_tester.run_simulation(scenarios, loan_data)

    st.subheader("Stress Test Results")

    # --- Display Results ---
    for i, scenario in enumerate(scenarios):
        st.subheader(f"Scenario {i+1}: {scenario['name']}") 

        # --- Display Scenario Parameters ---
        st.write(f"Interest Rate Change: {scenario['interest_rate_change']}")
        st.write(f"Unemployment Change: {scenario['unemployment_change']}")
        st.write(f"GDP Growth Change: {scenario['gdp_growth_change']}")
        st.write(f"Inflation Change: {scenario['inflation_change']}")
        # --- Visualization for scenario comparison ---
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=results[f'loss_{scenario["name"]}'], name=scenario["name"], histnorm='probability'))
        fig.update_layout(
            title_text='Distribution of Losses',
            xaxis_title='Loss Amount',
            yaxis_title='Probability',
            barmode='overlay'  # Enable overlay mode for comparison
        )
        fig.update_traces(opacity=0.75)
        st.plotly_chart(fig)

    # --- Display the stressed loan data (optional) ---
    st.subheader("Stressed Loan Data")
    st.write(results)