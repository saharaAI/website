import plotly.express as px
import pandas as pd
import random
import datetime
import streamlit as st

# --- Dashboard Styling --- 
st.set_page_config(layout="wide")  # Use wide layout for the dashboard

# Color Palette 
color_palette = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
]


class RiskMonitoringDashboard:
    def __init__(self):
        self.df = self.generate_simulated_data()

    def generate_simulated_data(self, num_records=500):
        """Generates simulated risk data with more features and customization."""
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
            "latitude": [random.uniform(30, 48) for _ in range(num_records)], 
            "longitude": [random.uniform(-125, -65) for _ in range(num_records)],
            "loan_amount": [random.randint(10000, 100000) for _ in range(num_records)],
            "interest_rate": [random.uniform(0.03, 0.15) for _ in range(num_records)],  # New feature
            "loan_term": [random.choice([12, 24, 36, 60]) for _ in range(num_records)],  # New feature
            "risk_category": random.choices(
                ["Low", "Medium", "High"], weights=[0.6, 0.3, 0.1], k=num_records
            ),
            "loan_purpose": random.choices(
                ["Auto", "Mortgage", "Personal", "Small Business"], weights=[0.2, 0.5, 0.2, 0.1], k=num_records
            ), # New feature
            "loan_id": [f"LN-{i+1:05}" for i in range(num_records)]
        }
        df = pd.DataFrame(data)

        # --- Add Trends ---
        df = df.sort_values(by="date")  # Sort by date to apply trends
        df["default_rate"] = df["default_rate"] * df.index.values * 0.00005 + df["default_rate"] 
        df["delinquency_rate"] = df["delinquency_rate"] * df.index.values * 0.00003 + df["delinquency_rate"]

        # --- Introduce Correlations --- 
        df["default_rate"] = df["default_rate"] - df["loan_amount"] * 0.00000005  
        df["default_rate"] = df["default_rate"].clip(0.01, 0.15) # Keep within reasonable bounds

        return df

    def display_interactive_plot(self):
        st.subheader("Interactive Risk Monitoring Dashboard")

        # --- 1. Key Risk Indicators Over Time ---
        selected_indicators = st.multiselect(
            "Select Indicators to View", 
            ["default_rate", "delinquency_rate", "loss_rate"],
            default=["default_rate", "delinquency_rate"] 
        )
        fig_time_series = px.line(
            self.df,
            x="date",
            y=selected_indicators,
            title="Key Risk Indicators Over Time"
        )
        fig_time_series.update_layout(xaxis_title="Date", yaxis_title="Rate (%)")
        st.plotly_chart(fig_time_series)

        # --- 2. Geographic Distribution of Risk ---
        st.subheader("Geographic Distribution of Loan Defaults")
        risk_metric = st.selectbox("Select Risk Metric for Map", ["default_rate", "loss_rate"])
        fig_map = px.scatter_geo(
            self.df,
            lat="latitude",
            lon="longitude",
            color=risk_metric,
            size="loan_amount",
            hover_name="loan_id",
            projection="natural earth",
            title=f"{risk_metric.capitalize()} by Location"
        )
        fig_map.update_layout(geo=dict(showcoastlines=True, coastlinewidth=0.5))
        st.plotly_chart(fig_map)

        # --- 3. Loan Amount Distribution by Risk Category ---
        st.subheader("Loan Amount Distribution by Risk Category")
        fig_box = px.box(
            self.df,
            x="risk_category",
            y="loan_amount",
            color="loan_purpose", # Color by loan purpose
            title="Loan Amount by Risk Category and Purpose"
        )
        fig_box.update_layout(
            xaxis_title="Risk Category", 
            yaxis_title="Loan Amount"
        )
        st.plotly_chart(fig_box)

        # --- 4. Default Rate by Loan Purpose (Bar Chart) ---
        st.subheader("Default Rate by Loan Purpose")
        fig_bar = px.bar(
            self.df.groupby("loan_purpose")["default_rate"].mean().reset_index(), 
            x="loan_purpose", 
            y="default_rate", 
            color="loan_purpose",
            title="Average Default Rate by Loan Purpose"
        )
        st.plotly_chart(fig_bar)

        # --- 5. Loan Term Distribution by Risk Category (Histogram) ---
        st.subheader("Loan Term Distribution by Risk Category")
        fig_hist = px.histogram(
            self.df,
            x="loan_term",
            color="risk_category",
            title="Distribution of Loan Terms by Risk Category",
            nbins=4
        )
        st.plotly_chart(fig_hist)


    
    def display_interactive_plot(self):
        st.title("Interactive Risk Monitoring Dashboard")

        # --- Layout: Two Columns ---
        col1, col2 = st.columns([2, 1]) 

        # --- Column 1: Time Series Chart ---
        with col1:
            selected_indicators = st.multiselect(
                "Select Indicators to View",
                ["default_rate", "delinquency_rate", "loss_rate"],
                default=["default_rate", "delinquency_rate"]
            )
            fig_time_series = px.line(
                self.df,
                x="date",
                y=selected_indicators,
                title="Key Risk Indicators Over Time",
                color_discrete_sequence=color_palette # Apply color palette
            )
            fig_time_series.update_layout(
                xaxis_title="Date", 
                yaxis_title="Rate (%)",
                plot_bgcolor="white", # White background for charts 
                xaxis=dict(showgrid=False), # Remove x-axis grid
                yaxis=dict(showgrid=True, gridcolor='lightgrey'), # Light grey grid
            )
            st.plotly_chart(fig_time_series, use_container_width=True)

        # --- Column 2: KPIs ---
        with col2:
            st.subheader("Key Performance Indicators")

            # Calculate KPIs - Replace with actual calculations 
            avg_default_rate = self.df["default_rate"].mean()
            total_loan_amount = self.df["loan_amount"].sum()
            high_risk_loans = len(self.df[self.df["risk_category"] == "High"])

            # Display KPIs with styling
            st.markdown(f"<h4 style='color:{color_palette[0]}; font-weight:bold;'>Average Default Rate</h4>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color:{color_palette[0]};'>{avg_default_rate:.2%}</h2>", unsafe_allow_html=True)

            st.markdown(f"<h4 style='color:{color_palette[1]}; font-weight:bold;'>Total Loan Amount</h4>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color:{color_palette[1]};'>${total_loan_amount:,.2f}</h2>", unsafe_allow_html=True)

            st.markdown(f"<h4 style='color:{color_palette[3]}; font-weight:bold;'>High Risk Loans</h4>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color:{color_palette[3]};'>{high_risk_loans}</h2>", unsafe_allow_html=True)


        # --- Layout: Two Columns (For the remaining charts) ---
        col3, col4 = st.columns(2)

        # --- Column 3: Geographic Map ---
        with col3:
            risk_metric = st.selectbox("Select Risk Metric for Map", ["default_rate", "loss_rate"])
            fig_map = px.scatter_geo(
                self.df,
                lat="latitude",
                lon="longitude",
                color=risk_metric,
                size="loan_amount",
                hover_name="loan_id",
                projection="natural earth",
                title=f"{risk_metric.capitalize()} by Location",
                color_continuous_scale="Reds"  # Use a color scale
            )
            fig_map.update_layout(
                geo=dict(showcoastlines=True, coastlinewidth=0.5), 
                plot_bgcolor="white" # White background for the map 
            )
            st.plotly_chart(fig_map, use_container_width=True)

        # --- Column 4: Loan Amount by Risk Category ---
        with col4:
            st.subheader("Loan Amount by Risk Category")
            fig_box = px.box(
                self.df,
                x="risk_category",
                y="loan_amount",
                color="loan_purpose",
                title="Loan Amount by Risk Category and Purpose",
                color_discrete_sequence=color_palette # Apply color palette
            )
            fig_box.update_layout(
                xaxis_title="Risk Category",
                yaxis_title="Loan Amount",
                plot_bgcolor="white"
            )
            st.plotly_chart(fig_box, use_container_width=True)

        # --- Full-Width Charts ---
        st.markdown("---") # Separator 

        # --- Default Rate by Loan Purpose (Bar Chart) ---
        fig_bar = px.bar(
            self.df.groupby("loan_purpose")["default_rate"].mean().reset_index(),
            x="loan_purpose",
            y="default_rate",
            color="loan_purpose",
            title="Average Default Rate by Loan Purpose",
            color_discrete_sequence=color_palette
        )
        fig_bar.update_layout(
            xaxis_title="Loan Purpose",
            yaxis_title="Default Rate (%)",
            plot_bgcolor="white",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='lightgrey'),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # --- Loan Term Distribution by Risk Category (Histogram) ---
        fig_hist = px.histogram(
            self.df,
            x="loan_term",
            color="risk_category",
            title="Distribution of Loan Terms by Risk Category",
            nbins=4, 
            color_discrete_sequence=color_palette
        )
        fig_hist.update_layout(
            xaxis_title="Loan Term (Months)",
            yaxis_title="Count",
            plot_bgcolor="white", 
            xaxis=dict(showgrid=False), 
            yaxis=dict(showgrid=True, gridcolor='lightgrey'),
        )
        st.plotly_chart(fig_hist, use_container_width=True)