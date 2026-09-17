# Supply Chain Risk Intelligence System

An end-to-end machine learning and analytics project to predict late delivery risk at the time of order placement, enabling logistics and operations teams to prioritize interventions before shipments leave the facility.

---

## 1. Overview
Delivery delays damage customer trust and drive up customer support costs. In real-world delivery and quick-commerce operations, finding out an order was delayed *after* the promised window has passed is too late. This project builds a machine learning pipeline and analytics suite to flag high-risk orders at checkout using only information known at order time.

## 2. Problem Statement
Given an e-commerce or retail order at placement, predict whether it has a high risk of late delivery (`Late_delivery_risk` = 1) versus arriving on time (`Late_delivery_risk` = 0). The goal is not just binary classification, but reliable probability risk scoring so dispatchers can triage limited intervention capacity (e.g., priority fulfillment, carrier reassignment).

## 3. Dataset
* **Source:** [DataCo Smart Supply Chain Dataset](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) (180,519 records, 53 initial attributes spanning 2015–2018).
* **Target:** `Late_delivery_risk` — 98,977 late (54.8%) vs. 81,542 on time (45.2%).
* **Data Hygiene:** Removed 100% null `Product Description` and 86.2% null `Order Zipcode`. Dropped PII (names, emails, passwords, street addresses) and IDs in [`notebooks/02_data_preprocessing.ipynb`](notebooks/02_data_preprocessing.ipynb).

## 4. Approach & Workflow
1. **Exploration & Cleaning:** Audit data quality, drop PII, and remove post-event leakage.
2. **Feature Engineering:** Stratified 80/20 train/test split **before** calculating grouped aggregates to guarantee zero leakage.
3. **Statistical Testing:** Formal hypothesis tests ($\chi^2$, t-test, ANOVA) to validate relationships.
4. **Model Benchmarking:** Train baseline and 5 classification algorithms on identical splits.
5. **Evaluation & Risk Tiers:** Convert predictions into 3 operational risk tiers and analyze decision thresholds.
6. **Error Analysis:** Break down false negatives across shipping modes, customer types, and regions.
7. **Forecasting & A/B Simulation:** Model order volumes over time and simulate operational intervention.
8. **Dashboard & SQL:** Interactive Streamlit app for real-time risk scoring and 24 analytical SQL queries.

## 5. EDA & Feature Engineering
* **Strict Leakage Prevention:** Removed `Days for shipping (real)`, `Delivery Status`, and `shipping date (DateOrders)` because they are only recorded after delivery ([`src/preprocessing.py`](src/preprocessing.py)).
* **High-Cardinality Management:** Dropped city/state columns (`Order City`, `Order State`, etc., which created 5,000+ sparse one-hot columns) while retaining `Market`, `Order Region`, and `Order Country`.
* **Engineered Signals:** Built `Urgent_Shipment` ($\le 2$ scheduled days), `Profit_Margin`, `Discount_Percentage`, `Sales_Per_Item`, and calendar features (`Order_Hour`, `Is_Weekend`) in [`src/features.py`](src/features.py).
* **Historical Aggregates:** Computed category and shipping mode frequency/sales aggregates on **training data only** (144,415 rows) and mapped them to test data (36,104 rows) to prevent data leakage ([`notebooks/03_feature_engineering.ipynb`](notebooks/03_feature_engineering.ipynb)).

## 6. SQL & Statistical Analysis
* **Chi-Square Test ($\chi^2$):** Tested `Shipping Mode` against `Late_delivery_risk` in [`notebooks/04_statistical_analysis.ipynb`](notebooks/04_statistical_analysis.ipynb). Result: $\chi^2 = 37,716.04$ ($p < 0.001$, $df = 3$), confirming shipping mode is non-randomly associated with delay risk.
* **T-Test:** Confirmed scheduled shipping days differed significantly between on-time and delayed orders ($p < 0.001$).
* **SQL Suite:** Wrote 24 queries across [`sql/business_analysis.sql`](sql/business_analysis.sql), [`sql/delivery_analysis.sql`](sql/delivery_analysis.sql), and [`sql/risk_analysis.sql`](sql/risk_analysis.sql) using CTEs, window functions (`RANK()`, `DENSE_RANK()`, `LAG()`), and cohort aggregations.

## 7. Machine Learning
All models were evaluated on the exact same 36,104 test holdout using a `ColumnTransformer` (median imputation + `StandardScaler` for numericals; most-frequent imputation + `OneHotEncoder` for categoricals) in [`notebooks/05_model_training.ipynb`](notebooks/05_model_training.ipynb).

## 8. Model Comparison
Actual verified test set results logged in [`data/processed/model_comparison.csv`](data/processed/model_comparison.csv):

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Classifier (Baseline)** | 0.548 | 0.548 | 1.000 | 0.708 | 0.500 | 0.548 |
| **Logistic Regression** | 0.725 | 0.881 | 0.577 | 0.698 | 0.776 | 0.842 |
| **Decision Tree** | 0.794 | 0.810 | **0.816** | **0.813** | 0.792 | 0.762 |
| **Random Forest** | 0.753 | 0.857 | 0.660 | 0.745 | 0.846 | 0.885 |
| **Extra Trees (Selected)** | **0.796** | 0.859 | 0.751 | 0.801 | **0.891** | **0.916** |
| **Hist Gradient Boosting** | 0.739 | **0.892** | 0.597 | 0.715 | 0.834 | 0.880 |

## 9. Final Model & Why It Was Selected
While a single Decision Tree scored a slightly higher binary F1 (0.813 vs. 0.801), individual tree leaves output uncalibrated probabilities (mostly 0 or 1). For an operational triage system, probability calibration is critical.
* **Extra Trees** achieved the highest discriminative power: **0.891 ROC-AUC** and **0.916 PR-AUC**.
* **Risk Categorization:** In [`notebooks/06_model_evaluation.ipynb`](notebooks/06_model_evaluation.ipynb), probabilities were bucketed into Low ($p < 0.30$), Medium ($0.30 \le p < 0.60$), and High ($p \ge 0.60$) risk.
* **Threshold Tuning:** Lowering the decision threshold from $0.50 \to 0.40$ raises recall to **82.5%**, capturing ~1,500 additional delayed orders at the expense of a modest increase in false alarms.

## 10. Explainability & Feature Importance
* **Tree Importance:** In [`images/Feature_Importance.png`](images/Feature_Importance.png), `Days for shipment (scheduled)` and `Shipping Mode` emerge as the primary predictive drivers, followed by order sales value and temporal factors.
* **SHAP Attribution:** In [`notebooks/08_explainability.ipynb`](notebooks/08_explainability.ipynb), Tree SHAP reveals directionality: Standard Class shipping consistently exerts positive attribution toward late risk, whereas higher scheduled days reduce predicted risk.

## 11. Error Analysis
In [`notebooks/07_error_analysis.ipynb`](notebooks/07_error_analysis.ipynb), breaking down performance across operational segments revealed:
* **First Class & Same Day:** F1 of **1.000** and **0.983** (almost perfectly predictable due to strict transit windows).
* **Standard Class:** F1 drops to **0.513** (Recall: 0.403). It represents ~60% of all orders and accounts for **over 95% of all false negatives** because its 4–6 day scheduled window introduces external logistics variance.
* **Segment Fairness:** F1 scores across Consumer (0.794), Corporate (0.806), and Home Office (0.814) segments show steady performance without demographic bias.

## 12. Forecasting & Experimentation Simulation
* **Order Volume Forecasting:** In [`notebooks/09_forecasting.ipynb`](notebooks/09_forecasting.ipynb), aggregated 1,127 daily order counts (2015–2018, average 160 orders/day) and fitted a 7-day seasonal Holt-Winters Exponential Smoothing model to anticipate operational dispatch spikes.
* **A/B Test Simulation:** In [`notebooks/10_ab_test_simulation.ipynb`](notebooks/10_ab_test_simulation.ipynb), designed a counterfactual simulation evaluating a 30% delay-reduction intervention on model-flagged high-risk orders, measuring statistically significant late-rate reductions via two-sample z-testing.

## 13. Streamlit Dashboard
An interactive dashboard in [`app/app.py`](app/app.py) provides:
1. **Big Picture Overview:** Test set metrics, operational risk tier distributions, and action matrices.
2. **Key Insights:** Interactive charts showing delay drivers across shipping modes and global markets.
3. **Live Predictor:** Real-time risk scoring form returning predicted probability, risk badge (Low/Medium/High), and human-friendly operational explanations.

## 14. Key Findings
1. **Shipping mode is the dominant bottleneck:** Standard Class is where delays actually happen; express modes are tightly monitored and reliably on time.
2. **Probabilities beat binary labels:** Operations teams can triage intervention by sorting orders by risk score rather than treating all flags equally.
3. **Scheduled days dictate predictability:** The tighter the delivery window, the more deterministic the outcome; wider delivery windows introduce unobserved transit noise.

## 15. Limitations
* **Observational Historical Data:** Uses Kaggle retail supply chain data; does not include live telemetry, GPS coordinates, or real-time traffic/weather data.
* **Standard Class Uncertainty:** Tabular order-time features alone cannot capture in-transit vehicle breakdowns or courier handoff friction.
* **Simulated Experimentation:** The A/B test is a counterfactual statistical simulation, not a live production experiment.

## 16. Tech Stack
* **Language & Core:** Python, Pandas, NumPy
* **Machine Learning:** Scikit-learn (Pipelines, ExtraTrees, RandomForest, HistGradientBoosting)
* **Statistics & Forecasting:** SciPy (Chi-Square, T-Test), Statsmodels (Holt-Winters)
* **Explainability & Viz:** SHAP, Matplotlib, Seaborn
* **Application & SQL:** Streamlit, SQLite

## 17. How to Run
```bash
# 1. Clone repository and install requirements
git clone https://github.com/YogeshKhillare04/supply-chain-risk-intelligence.git
cd supply-chain-risk-intelligence
pip install -r requirements.txt

# 2. Download dataset (see data/README.md) into data/raw/DataCoSupplyChainDataset.csv

# 3. Launch Streamlit dashboard
streamlit run app/app.py

# 4. Explore notebooks in sequence (01 to 10)
jupyter notebook notebooks/
```
