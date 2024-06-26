import pandas as pd
import numpy as np
from scipy.stats import norm
import streamlit as st
import plotly.graph_objects as go

class StressTester:
    def __init__(self):
        pass

    def run_simulation(self, economic_scenarios, loan_data):
        """Executes stress tests for multiple economic scenarios."""
        results = []

        loan_data = loan_data.copy()
        if 'interest_rate' not in loan_data.columns:
            loan_data['interest_rate'] = 0.0

        for scenario in economic_scenarios:
            scenario_results = loan_data.copy()
            scenario_results['stressed_interest_rate'] = loan_data['interest_rate'] + scenario['interest_rate_change']

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
            results.append(scenario_results[['loan_amount', 'interest_rate', 'credit_score', f'default_{scenario["name"]}', f'loss_{scenario["name"]}']])

        combined_results = pd.concat(results, axis=1)
        return combined_results

    def simulate_default_probability(self, credit_score, loan_amount, interest_rate, macro_change):
        """Simulates default probability based on several risk factors."""
        weights = {
            'credit_score': -0.005,
            'loan_amount': 0.000001,
            'interest_rate': 0.2,
            'macro_change': 0.03
        }
        risk_score = (
            weights['credit_score'] * credit_score +
            weights['loan_amount'] * loan_amount +
            weights['interest_rate'] * interest_rate +
            weights['macro_change'] * macro_change
        )
        default_probability = norm.cdf(risk_score)
        return max(0, min(default_probability, 1))

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
base_params = {
    'interest_rate_change': st.slider("Base Interest Rate Change (%)", -5.0, 5.0, 0.0, 0.25) / 100,
    'unemployment_change': st.slider("Base Unemployment Rate Change (%)", -10.0, 10.0, 0.0, 0.5) / 100,
    'gdp_growth_change': st.slider("Base GDP Growth Change (%)", -5.0, 5.0, 0.0, 0.25) / 100,
    'inflation_change': st.slider("Base Inflation Change (%)", -5.0, 5.0, 0.0, 0.25) / 100
}

# Create a list to store the scenarios
scenarios = []
num_scenarios = st.number_input("Number of scenarios", min_value=1, max_value=5, value=2)
for i in range(num_scenarios):
    st.markdown(f"**Scenario {i+1} Adjustments:**")
    scenario_params = {
        'name': f'Scenario {i+1}',
        'interest_rate_change': base_params['interest_rate_change'] + st.slider(f"Interest Rate Adjustment (Scenario {i+1}) (%)", -5.0, 5.0, 0.0, 0.25) / 100,
        'unemployment_change': base_params['unemployment_change'] + st.slider(f"Unemployment Rate Adjustment (Scenario {i+1}) (%)", -10.0, 10.0, 0.0, 0.5) / 100,
        'gdp_growth_change': base_params['gdp_growth_change'] + st.slider(f"GDP Growth Adjustment (Scenario {i+1}) (%)", -5.0, 5.0, 0.0, 0.25) / 100,
        'inflation_change': base_params['inflation_change'] + st.slider(f"Inflation Adjustment (Scenario {i+1}) (%)", -5.0, 5.0, 0.0, 0.25) / 100
    }
    scenarios.append(scenario_params)

# --- Instantiate and Run Stress Test ---
stress_tester = StressTester()

if st.button("Run Stress Test"):
    results = stress_tester.run_simulation(scenarios, loan_data)

    st.subheader("Stress Test Results")

    # --- Display Results ---
    for i, scenario in enumerate(scenarios):
        st.subheader(f"Scenario {i+1}: {scenario['name']}") 
        st.write(scenario)

        # --- Visualization for scenario comparison ---
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=results[f'loss_{scenario["name"]}'], name=scenario["name"], histnorm='probability'))
        fig.update_layout(
            title_text='Distribution of Losses',
            xaxis_title='Loss Amount',
            yaxis_title='Probability',
            barmode='overlay'
        )
        fig.update_traces(opacity=0.75)
        st.plotly_chart(fig)

    # --- Display the stressed loan data (optional) ---
    st.subheader("Stressed Loan Data")
    st.write(results)
