import plotly.express as px
import pandas as pd
import random
import datetime
import streamlit as st

class RiskMonitoringDashboard:
    def __init__(self):
        self.df = self.generate_simulated_data()

    def generate_simulated_data(self, num_records=100):
        """Generates simulated risk data."""
        start_date = datetime.date(2022, 1, 1) 
        end_date = datetime.date(2023, 12, 31)

        data = {
            "date": [
                start_date + datetime.timedelta(days=random.randint(0, (end_date - start_date).days)) 
                for _ in range(num_records)
            ],
            "default_rate": [random.uniform(0.01, 0.08) for _ in range(num_records)],
            "delinquency_rate": [random.uniform(0.05, 0.15) for _ in range(num_records)],
            "loss_rate": [random.uniform(0.005, 0.03) for _ in range(num_records)],
            "latitude": [random.uniform(30, 48) for _ in range(num_records)],  # Approximate US latitudes
            "longitude": [random.uniform(-125, -65) for _ in range(num_records)], # Approximate US longitudes
            "loan_amount": [random.randint(10000, 100000) for _ in range(num_records)],
            "risk_category": random.choices(
                ["Low", "Medium", "High"], weights=[0.6, 0.3, 0.1], k=num_records
            ), 
            "loan_id": [f"LN-{i+1:05}" for i in range(num_records)] # Example unique loan IDs 
        }
        df = pd.DataFrame(data)

        # --- Customize Data Generation (Optional) ---
        # 1.  Trends: Add trends to the data (e.g., increasing default rates).
        # 2. Correlations: Introduce correlations (e.g., higher loan amounts might
        #     have lower default rates).
        # 3. Geographic Clusters: Create geographic clusters of higher or lower risk.

        return df

    # ... (Rest of the RiskMonitoringDashboard code from the previous example) ...
    def display_interactive_plot(self):
        """Displays the interactive risk monitoring dashboard."""

        # --- Example 1: Time Series of Key Risk Indicators ---
        st.subheader("Key Risk Indicators Over Time")
        fig_time_series = px.line(
            self.df, 
            x="date",  # Replace 'date' with your actual date column
            y=["default_rate", "delinquency_rate", "loss_rate"],  # Replace with your actual columns
            title="Key Risk Indicators",
        )
        fig_time_series.update_layout(xaxis_title="Date", yaxis_title="Rate (%)")
        st.plotly_chart(fig_time_series)

        # --- Example 2: Geographic Distribution of Risk ---
        st.subheader("Geographic Distribution of Loan Defaults")
        fig_map = px.scatter_geo(
            self.df,
            lat="latitude",  # Replace with your actual latitude column
            lon="longitude",  # Replace with your actual longitude column
            color="default_rate",  # Replace with a relevant risk indicator
            size="loan_amount",  # Replace with a relevant size variable 
            hover_name="loan_id",  # Replace with a unique identifier
            projection="natural earth", 
            title="Loan Defaults by Location",
        )
        fig_map.update_layout(geo=dict(showcoastlines=True, coastlinewidth=0.5))
        st.plotly_chart(fig_map)

        # --- Example 3: Distribution of Loan Amounts by Risk Category ---
        st.subheader("Loan Amounts by Risk Category")
        fig_box = px.box(
            self.df, 
            x="risk_category",  # Replace with your risk category column
            y="loan_amount", 
            title="Loan Amount Distribution by Risk Category",
        )
        fig_box.update_layout(xaxis_title="Risk Category", yaxis_title="Loan Amount")
        st.plotly_chart(fig_box)

        # ... Add more visualizations as needed ... 