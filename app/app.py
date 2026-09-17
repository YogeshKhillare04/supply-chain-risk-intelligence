"""
Supply Chain Risk Intelligence — Streamlit Dashboard

A clean, human-friendly dashboard for exploring delivery risk predictions 
and understanding what causes delivery delays.

Run locally with: streamlit run app/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.evaluation import get_risk_category

# Set consistent styling
plt.style.use('seaborn-v0_8-whitegrid')


# --- Page Configuration ---
st.set_page_config(
    page_title="Supply Chain Risk Intelligence",
    page_icon="📦",
    layout="wide"
)


# --- Helper Functions (Cached for speed) ---
@st.cache_data
def load_data():
    """Load test dataset with high-cardinality location fields dropped."""
    try:
        X_test = pd.read_csv("data/processed/X_test.csv")
        y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
        high_card_cols = ["Order City", "Order State", "Customer City", "Customer State"]
        X_test = X_test.drop(columns=[c for c in high_card_cols if c in X_test.columns])
        return X_test, y_test
    except FileNotFoundError:
        return None, None


@st.cache_resource
def load_model():
    """Load trained ExtraTrees pipeline."""
    try:
        model = joblib.load("models/best_model.pkl")
        return model
    except FileNotFoundError:
        return None


@st.cache_data
def get_cached_predictions(_model, X_test):
    """Cache probabilities so tab switching is instantaneous."""
    return _model.predict_proba(X_test)[:, 1]


# --- App Header ---
st.title("📦 Supply Chain Risk Intelligence")
st.markdown(
    "A practical machine learning project that predicts whether an order is at risk "
    "of being delayed **at checkout time** — giving operations teams time to take action."
)

with st.expander("ℹ️ Project Background & Context (Why We Built This)"):
    st.markdown("""
    * **The Problem:** In delivery operations (like Swiggy, Zomato, or e-commerce), discovering an order is late *after* it happens is too late. Customers get unhappy and support costs spike.
    * **Our Approach:** By analyzing order characteristics (shipping method, scheduled days, items, destination) the moment an order is placed, we assign an **actionable risk score**.
    * **Zero Data Leakage:** We strictly excluded any information that is only known after delivery (like actual transit time or dispatch timestamps).
    * **The Model:** Trained on **180,000+ real-world orders** using an `ExtraTreesClassifier` that achieves an **ROC-AUC of 0.891**.
    """)


# --- Load Resources ---
X_test, y_test = load_data()
model = load_model()

if X_test is None or model is None:
    st.error("⚠️ Data or model file not found. Make sure `data/processed/X_test.csv` and `models/best_model.pkl` exist.")
    st.stop()

# Get cached predictions for the 36,104 test orders
y_prob = get_cached_predictions(model, X_test)
risk_cats = get_risk_category(y_prob)


# --- Sidebar Navigation ---
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📊 Big Picture Overview", "🔍 Key Insights & Patterns", "🎯 Interactive Order Predictor"]
)

st.sidebar.divider()
st.sidebar.markdown("### About This App")
st.sidebar.info(
    "**Author:** Final Year Data Science Project\n\n"
    "**Tech:** Python, Scikit-learn, Streamlit\n\n"
    "**Core Metric:** ROC-AUC: **0.891** | F1: **0.801**\n\n"
    "**Dataset:** DataCo Global Supply Chain"
)


# ============================================================
# TAB 1: Big Picture Overview
# ============================================================
if page == "📊 Big Picture Overview":
    st.subheader("How the Model Performs on Unseen Orders")
    st.markdown("Here is how our model evaluated the **36,104 test orders** it had never seen during training:")

    # Top high-level KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Test Orders Evaluated", f"{len(y_test):,}")
    with kpi2:
        st.metric("Actual Late Delivery Rate", f"{y_test.mean() * 100:.1f}%")
    with kpi3:
        st.metric("Model ROC-AUC Score", "0.891", help="Shows strong ability to separate late orders from on-time orders")
    with kpi4:
        st.metric("Model Accuracy", "79.6%")

    st.divider()

    st.markdown("### Why We Group Orders into Risk Tiers")
    st.write(
        "In real operations, warehouse managers don't have time to inspect thousands of binary '0 vs 1' predictions. "
        "Instead, we group orders into **Actionable Risk Categories** so dispatchers know who to prioritize first:"
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        # Clean risk category bar chart
        risk_counts = pd.Series(risk_cats).value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"])
        
        fig, ax = plt.subplots(figsize=(6, 3.8))
        colors = ["#2ecc71", "#f39c12", "#e74c3c"]
        bars = ax.bar(risk_counts.index, risk_counts.values, color=colors, alpha=0.85, width=0.5)
        ax.set_ylabel("Number of Orders")
        ax.set_title("Order Volume by Risk Tier", fontsize=12, fontweight='bold')
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height):,}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
            
        st.pyplot(fig)
        plt.close()

    with col2:
        # Human breakdown table
        st.markdown("#### Operational Action Matrix")
        summary_data = []
        for cat, desc, action in [
            ("Low Risk", "Probability < 30%", "Standard fulfillment. No intervention required."),
            ("Medium Risk", "Probability 30% – 60%", "Watchlist. Automated notification sent to carrier hub."),
            ("High Risk", "Probability > 60%", "Priority dispatch. Assign backup delivery partner or expedite.")
        ]:
            count = (risk_cats == cat).sum()
            actual_late = y_test[risk_cats == cat].mean() * 100
            summary_data.append({
                "Risk Tier": cat,
                "Definition": desc,
                "Orders": f"{count:,}",
                "Actual Delay Rate": f"{actual_late:.1f}%",
                "Recommended Action": action
            })
        
        st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)


# ============================================================
# TAB 2: Key Insights & Patterns
# ============================================================
elif page == "🔍 Key Insights & Patterns":
    st.subheader("What Drives Delivery Delays?")
    st.markdown(
        "Here are the biggest operational lessons learned from our exploratory data analysis, "
        "statistical tests, and error analysis."
    )

    insight_tab = st.radio(
        "Choose an Area to Explore:",
        ["1. Shipping Method (The Biggest Factor)", "2. Geographic Markets", "3. Top Predictive Features"],
        horizontal=True
    )

    analysis_df = X_test.copy()
    analysis_df["Actual_Late"] = y_test.values

    if "1. Shipping Method" in insight_tab:
        st.markdown("#### Insight #1: Shipping Mode Explains Almost Everything")
        st.write(
            "Our Chi-Square test confirmed a massive relationship between shipping mode and delay risk (p < 0.001). "
            "Here is the real breakdown:"
        )

        ship_stats = analysis_df.groupby("Shipping Mode")["Actual_Late"].agg(["count", "mean"]).reset_index()
        ship_stats.columns = ["Shipping Mode", "Order Count", "Delay Rate"]
        ship_stats["Delay Rate %"] = (ship_stats["Delay Rate"] * 100).round(1)
        ship_stats = ship_stats.sort_values("Delay Rate %", ascending=False)

        col1, col2 = st.columns([1, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            sns.barplot(data=ship_stats, x="Shipping Mode", y="Delay Rate %", palette="Blues_r", ax=ax)
            ax.set_ylabel("Delay Rate (%)")
            ax.set_title("Delay Rate by Shipping Mode", fontweight='bold')
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("""
            * **First Class & Same Day:** Almost 100% predictable because their transit windows are tightly scheduled.
            * **Standard Class:** Represents **60%+ of all orders** and has a **40%+ delay rate**. Because delivery windows span 4–6 days, external transit variability causes ~95% of all prediction errors.
            * **Actionable Advice:** Operations should focus intervention protocols primarily on **Standard Class** shipments.
            """)

    elif "2. Geographic Markets" in insight_tab:
        st.markdown("#### Insight #2: Delays Across Global Destinations")
        market_stats = analysis_df.groupby("Market")["Actual_Late"].agg(["count", "mean"]).reset_index()
        market_stats.columns = ["Market", "Order Count", "Delay Rate"]
        market_stats["Delay Rate %"] = (market_stats["Delay Rate"] * 100).round(1)
        market_stats = market_stats.sort_values("Delay Rate %", ascending=False)

        col1, col2 = st.columns([1, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            sns.barplot(data=market_stats, x="Market", y="Delay Rate %", palette="Purples_r", ax=ax)
            ax.set_ylabel("Delay Rate (%)")
            ax.set_title("Delay Rate by Market", fontweight='bold')
            ax.tick_params(axis="x", rotation=25)
            st.pyplot(fig)
            plt.close()

        with col2:
            st.dataframe(market_stats[["Market", "Order Count", "Delay Rate %"]], hide_index=True, use_container_width=True)
            st.caption("Notice that delay rates are fairly consistent (52%–57%) across all major markets, showing that logistics bottlenecks are mostly carrier-driven rather than country-specific.")

    elif "3. Top Predictive Features" in insight_tab:
        st.markdown("#### Insight #3: What Does the Model Pay Most Attention To?")
        
        try:
            feature_names = model.named_steps["preprocessor"].get_feature_names_out()
            importances = model.named_steps["classifier"].feature_importances_
            feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
            top10 = feat_imp.sort_values("Importance", ascending=False).head(10)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=top10, x="Importance", y="Feature", palette="viridis", ax=ax)
            ax.set_title("Top 10 Most Important Features in ExtraTrees Model", fontweight='bold')
            st.pyplot(fig)
            plt.close()
            st.info("💡 **Takeaway:** Scheduled shipping days, shipping mode, and order financial value rank highest. Date of order and category also provide marginal lifts.")
        except Exception:
            st.write("Feature importance preview currently available in Notebook 05 & 06.")


# ============================================================
# TAB 3: Interactive Order Predictor
# ============================================================
elif page == "🎯 Interactive Order Predictor":
    st.subheader("Test an Order in Real Time")
    st.markdown(
        "Try adjusting the order parameters below to see how the model evaluates its late delivery risk in real time:"
    )

    sample = X_test.iloc[0].copy()

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### 🚚 Shipping & Timeline")
            shipping_mode = st.selectbox(
                "Shipping Option Chosen by Customer",
                ["Standard Class", "Second Class", "First Class", "Same Day"],
                help="Standard Class has historically higher variance"
            )
            scheduled_days = st.slider(
                "Scheduled Delivery Days Promised",
                min_value=1, max_value=6, value=4,
                help="Promised timeline at checkout"
            )
            order_quantity = st.number_input(
                "Item Quantity in Order",
                min_value=1, max_value=10, value=1
            )

        with col2:
            st.markdown("##### 📍 Destination & Value")
            market = st.selectbox(
                "Destination Market",
                ["USCA", "Europe", "Pacific Asia", "LATAM", "Africa"]
            )
            sales_value = st.number_input(
                "Order Total ($ USD)",
                min_value=10.0, max_value=2000.0, value=180.0, step=20.0
            )
            customer_segment = st.selectbox(
                "Customer Type",
                ["Consumer", "Corporate", "Home Office"]
            )

        submit_btn = st.form_submit_button("⚡ Evaluate Delivery Risk", type="primary", use_container_width=True)

    if submit_btn:
        input_row = sample.copy()
        
        # Map user choices into input row
        input_row["Shipping Mode"] = shipping_mode
        input_row["Days for shipment (scheduled)"] = scheduled_days
        input_row["Order Item Quantity"] = order_quantity
        input_row["Market"] = market
        input_row["Sales"] = sales_value
        input_row["Customer Segment"] = customer_segment
        if "Urgent_Shipment" in input_row.index:
            input_row["Urgent_Shipment"] = 1 if scheduled_days <= 2 else 0

        input_df = pd.DataFrame([input_row])

        try:
            prob = model.predict_proba(input_df)[0, 1]
            risk_tier = get_risk_category(prob)

            st.divider()
            st.markdown("### Prediction Outcome")

            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.metric("Estimated Risk of Delay", f"{prob:.1%}")
            with res_col2:
                badge = {"Low Risk": "🟢 Low Risk", "Medium Risk": "🟡 Medium Risk", "High Risk": "🔴 High Risk"}
                st.metric("Assigned Category", badge.get(risk_tier, risk_tier))
            with res_col3:
                decision = "Likely Delayed" if prob >= 0.5 else "Likely On Time"
                st.metric("Expected Outcome", decision)

            st.progress(min(float(prob), 1.0))

            # Friendly human explanation
            st.markdown("#### 💬 Plain English Explanation")
            if risk_tier == "Low Risk":
                st.success(
                    f"**Low Risk ({prob:.1%}):** This order looks very safe. Given the '{shipping_mode}' mode and {scheduled_days} scheduled days, "
                    "fulfillment can proceed through standard warehouse queues without manual intervention."
                )
            elif risk_tier == "Medium Risk":
                st.warning(
                    f"**Medium Risk ({prob:.1%}):** This order is on the border. While not guaranteed to be late, any initial delay at the packing hub "
                    "could push it past the promised date. Recommended to monitor hub departure."
                )
            else:
                st.error(
                    f"**High Risk ({prob:.1%}):** High likelihood of delay. The combination of '{shipping_mode}' with {scheduled_days} scheduled days "
                    "creates a tight operational window. Recommend prioritizing packing or assigning an expedited partner."
                )

        except Exception as err:
            st.error(f"Prediction error: {err}")
