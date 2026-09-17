# Supply Chain Risk Intelligence: Predicting Late Deliveries at Checkout

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end data science project predicting late delivery risk at the exact moment of order placement—giving operations and logistics teams the lead time needed to intervene before packages leave the facility.

---

## The Problem: Late Deliveries at Scale
In supply chain and e-commerce operations, finding out a delivery is delayed after the promised window has passed is useless for prevention. Once an order is loaded onto a long-haul truck, dispatchers cannot easily reroute or prioritize it.

The objective of this project is to build an operational risk-scoring system that flags delayed orders (`Late_delivery_risk` = 1) at **order checkout**, using only attributes available before fulfillment. Rather than relying on a static binary label, the pipeline scores calibrated probabilities so warehouse managers can triage limited expediting capacity (priority packing, carrier reassignment) toward orders that need it most.

## Dataset & The Leakage Trap
I used the [DataCo Smart Supply Chain Dataset](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis), which tracks 180,519 order shipments across global markets between 2015 and 2018.
- **Target Distribution:** 98,977 late orders (54.8%) vs. 81,542 on-time orders (45.2%).
- **Data Hygiene:** Dropped customer PII (names, emails, passwords, street addresses) and redundant IDs. Removed `Product Description` (100% missing) and `Order Zipcode` (86.2% missing).
- **Preventing Target Leakage:** The most critical step in [`notebooks/02_data_preprocessing.ipynb`](notebooks/02_data_preprocessing.ipynb) was eliminating post-event leakage columns. Features like `Days for shipping (real)`, `Delivery Status`, and `shipping date (DateOrders)` are recorded after shipment. Leaving them in produces an artificial ~99% accuracy model that is useless in real life.
- **High Cardinality:** Dropping raw city and state columns (`Order City`, `Customer City`, etc.) prevented 5,000+ sparse dummy columns, keeping reliable regional signals (`Market`, `Order Region`, `Order Country`).

## Workflow & Feature Engineering
To ensure strict separation, I split the 180,519 records into an 80% train set (144,415) and 20% test set (36,104) with stratification **before** engineering aggregate features:
1. **Engineered Domain Signals:** In [`src/features.py`](src/features.py), I extracted delivery urgency (`Urgent_Shipment` $\le$ 2 scheduled days), financial signals (`Profit_Margin`, `Discount_Percentage`, `Sales_Per_Item`), and checkout timing (`Order_Hour`, `Is_Weekend`).
2. **Out-of-Fold Aggregates:** Computed category and shipping-tier order volume aggregations strictly on the training partition and mapped them onto the test set to avoid lookahead bias ([`notebooks/03_feature_engineering.ipynb`](notebooks/03_feature_engineering.ipynb)).

## Statistical Testing & SQL Analytics
Before jumping into machine learning, I ran formal hypothesis tests in [`notebooks/04_statistical_analysis.ipynb`](notebooks/04_statistical_analysis.ipynb):
- **Chi-Square Test ($\chi^2$):** Evaluated `Shipping Mode` against `Late_delivery_risk`. The test yielded $\chi^2 = 37,716.04$ ($p < 0.001$, $df = 3$), statistically confirming that shipping mode is strongly tied to delay outcomes.
- **Two-Sample T-Test:** Confirmed scheduled transit days between delayed and on-time shipments differ significantly ($p < 0.001$).
- **SQL Analysis:** Implemented 24 analytical queries across [`sql/business_analysis.sql`](sql/business_analysis.sql), [`sql/delivery_analysis.sql`](sql/delivery_analysis.sql), and [`sql/risk_analysis.sql`](sql/risk_analysis.sql) using CTEs, window functions (`RANK()`, `LAG()`), and cohort aggregations to examine route bottlenecks and customer lifetime revenue.

## Model Benchmarking
I evaluated 5 classifiers against a naive Dummy Baseline using a standardized `ColumnTransformer` (median imputation + scaling for continuous variables, frequent imputation + one-hot encoding for categoricals). All models were scored on the exact same 36,104 test records ([`data/processed/model_comparison.csv`](data/processed/model_comparison.csv)):

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Classifier (Baseline)** | 0.548 | 0.548 | 1.000 | 0.708 | 0.500 | 0.548 |
| **Logistic Regression** | 0.725 | 0.881 | 0.577 | 0.698 | 0.776 | 0.842 |
| **Decision Tree** | 0.794 | 0.810 | **0.816** | **0.813** | 0.792 | 0.762 |
| **Random Forest** | 0.753 | 0.857 | 0.660 | 0.745 | 0.846 | 0.885 |
| **Extra Trees (Selected)** | **0.796** | 0.859 | 0.751 | 0.801 | **0.891** | **0.916** |
| **Hist Gradient Boosting** | 0.739 | **0.892** | 0.597 | 0.715 | 0.834 | 0.880 |

## Why Extra Trees Was Selected
On paper, a single Decision Tree scored a slightly higher default F1 (0.813 vs. 0.801). However, single decision trees produce uncalibrated, polarized probabilities (almost entirely 0 or 1). 

In an operational setting, dispatch teams cannot act on every single order; they need to rank orders from riskiest to safest. **Extra Trees** was selected as the final production model because:
1. It delivered the highest ranking discrimination: **0.891 ROC-AUC** and **0.916 PR-AUC**.
2. Its smooth probability distribution enabled clean operational tiering: **Low** ($p < 0.30$), **Medium** ($0.30 \le p < 0.60$), and **High** ($p \ge 0.60$) risk ([`notebooks/06_model_evaluation.ipynb`](notebooks/06_model_evaluation.ipynb)).
3. Tuning the operational threshold from $0.50 \to 0.40$ raises recall to **82.5%**, flagging ~1,500 additional delayed shipments with minimal extra false alarms.

## Explainability & What Drives Delays
Using Tree SHAP and Gini impurity feature importances, the primary drivers of late delivery risk became clear:

<p align="center">
  <img src="images/Feature_Importance.png" alt="Top 20 Feature Importances" width="800"/>
</p>

- `Shipping Mode (Standard Class)` and `Urgent_Shipment` ($\le 2$ days scheduled) are the top predictive features.
- SHAP values in [`notebooks/08_explainability.ipynb`](notebooks/08_explainability.ipynb) show that Standard Class pushes risk scores up significantly, while longer scheduled transit buffers consistently push risk scores down.

## Error Analysis: Where Does The Model Struggle?
In [`notebooks/07_error_analysis.ipynb`](notebooks/07_error_analysis.ipynb), I segmented model performance by shipping method:
- **First Class & Same Day:** F1 scores are **1.000** and **0.983**. These express tiers are virtually deterministic because carriers adhere to strict delivery windows.
- **Standard Class:** F1 drops to **0.513** (Recall: 0.403). Standard Class accounts for ~60% of all volume and **over 95% of all false negatives**. With a wide 4 to 6 day scheduled window, whether a package arrives on day 4 or day 5 depends on real-world transit delays (traffic, weather, hub congestion) that cannot be observed from checkout data alone.
- **Demographic Parity:** Performance across Consumer (F1 0.794), Corporate (0.806), and Home Office (0.814) segments is balanced, confirming no segment-level performance bias.

## Forecasting & A/B Test Simulation
- **Daily Volume Forecasting:** In [`notebooks/09_forecasting.ipynb`](notebooks/09_forecasting.ipynb), I aggregated 1,127 days of order history (averaging ~160 orders/day) and fitted a 7-day seasonal Holt-Winters Exponential Smoothing model to help dispatchers forecast staffing demand.
- **Simulated A/B Intervention:** In [`notebooks/10_ab_test_simulation.ipynb`](notebooks/10_ab_test_simulation.ipynb), I ran a counterfactual experiment simulating a 30% delay-reduction intervention applied to high-risk flagged orders. A two-sample z-test confirmed statistically significant delay rate reduction ($p < 0.001$).

## Streamlit Dashboard
The repository includes a live Streamlit dashboard in [`app/app.py`](app/app.py):
- **Executive Summary:** Test set benchmark cards, confusion matrices, and risk tier breakdowns.
- **Operational Insights:** Delay heatmaps across shipping methods and regional destination markets.
- **Live Risk Predictor:** An interactive form allowing an operator to input order details, receiving a calibrated risk score, Low/Medium/High risk badge, and a plain-English explanation of the risk factors.

## Honest Limitations
1. **Tabular Observational Data:** The dataset lacks live GPS telemetry, transit weather reports, and carrier handoff timestamps.
2. **Standard Class Noise:** As shown in error analysis, predicting Standard Class delays from checkout attributes alone has an inherent ceiling without real-time tracking data.
3. **Simulated Interventions:** The A/B test is a counterfactual statistical simulation rather than an in-vivo field trial.

## Tech Stack
- **Data & Modeling:** Python, Pandas, NumPy, Scikit-learn (Pipelines, ExtraTrees, HistGradientBoosting)
- **Inference & Explainability:** SHAP, Matplotlib, Seaborn
- **Statistical Testing & Time Series:** SciPy (Stats), Statsmodels (Holt-Winters)
- **Deployment & Querying:** Streamlit, SQLite

## How to Run
```bash
# 1. Clone the repository
git clone https://github.com/YogeshKhillare04/supply-chain-risk-intelligence.git
cd supply-chain-risk-intelligence

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add the raw dataset (follow data/README.md)
# Place DataCoSupplyChainDataset.csv inside data/raw/

# 4. Launch the Streamlit dashboard
streamlit run app/app.py

# 5. Run the notebooks in numerical order (01 to 10)
jupyter notebook notebooks/
```
