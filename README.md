# 🚚 Supply Chain Risk Intelligence System

Predict shipment delivery risk at order checkout using Machine Learning to help logistics and operations teams identify high-risk shipments before dispatch.

---

## 📌 Overview

This project develops an end-to-end Machine Learning and analytics pipeline to predict **Late Delivery Risk** (`Late_delivery_risk` = 1) using the **DataCo Smart Supply Chain Dataset (180,519 records)**. 

Finding out an order is delayed after the promised window has passed is too late for logistics intervention. This system scores calibrated delay probabilities at **order placement**, allowing warehouse dispatchers to triage limited expediting capacity (priority packaging, carrier reassignment) before packages leave the facility.

---

## 🚀 Key Highlights & Capabilities

- 🧹 **Strict Leakage-Free Preprocessing:** Purged post-event columns (`Days for shipping (real)`, `Delivery Status`, and ship dates) to prevent artificial 99% accuracy.
- ⚙️ **Production Feature Engineering:** Engineered shipment urgency flags, financial margin ratios, order timing, and leak-free out-of-fold training aggregations.
- 📊 **Statistical Testing & SQL Suite:** Validated delay drivers with Chi-Square tests (Chi-Square = 37,716.04, p < 0.001) and 24 analytical SQL queries across routing lanes and customer cohorts.
- 🤖 **6 ML Models Benchmarked:** Compared Dummy baseline, Logistic Regression, Decision Tree, Random Forest, Extra Trees, and HistGradientBoosting on identical test splits.
- 🌲 **Calibrated Risk Scoring:** Extra Trees selected for best-in-class probability ranking (**0.891 ROC-AUC**, **0.916 PR-AUC**) and segmented into actionable Low, Medium, and High risk tiers.
- 🔍 **Explainability & SHAP:** Global feature importances and Tree SHAP values isolate exact directional delay drivers.
- 🎯 **Deep Error Analysis:** Uncovered why Standard Class shipping accounts for >95% of false negatives due to 4–6 day scheduled window variance.
- 📈 **Forecasting & A/B Simulation:** 7-day seasonal Holt-Winters order volume forecasting and counterfactual A/B intervention testing.
- 🖥️ **Interactive Streamlit App:** Live checkout risk predictor with instant probability scoring and operational recommendations.

---

## 📂 Project Workflow

```
DataCo Raw Dataset (180,519 orders)
               │
               ▼
Data Cleaning & Leakage Elimination (drop post-event columns & PII)
               │
               ▼
Stratified 80/20 Train/Test Split (144,415 train / 36,104 test)
               │
               ▼
Feature Engineering & Aggregations (urgency flags, margins, temporal signals)
               │
               ▼
Statistical Hypothesis Testing (Chi-Square & T-Test) + 24 SQL Queries
               │
               ▼
Model Training & Benchmarking (Baseline, Logistic, Trees, Ensembles, Boosting)
               │
               ▼
Operational Evaluation (Calibrated Risk Tiers, Threshold Tuning, Error Breakdown)
               │
               ▼
Explainability (SHAP & Feature Importance) + Forecasting + Streamlit Dashboard
```

---

## 📊 Models Compared & Test Results

All models were evaluated on the exact same 36,104 test holdout using a standardized `ColumnTransformer`:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Dummy Baseline | 0.548 | 0.548 | 1.000 | 0.708 | 0.500 | 0.548 |
| Logistic Regression | 0.725 | 0.881 | 0.577 | 0.698 | 0.776 | 0.842 |
| Decision Tree | 0.794 | 0.810 | **0.816** | **0.813** | 0.792 | 0.762 |
| Random Forest | 0.753 | 0.857 | 0.660 | 0.745 | 0.846 | 0.885 |
| **Extra Trees (Selected)** | **0.796** | 0.859 | 0.751 | 0.801 | **0.891** | **0.916** |
| Hist Gradient Boosting | 0.739 | **0.892** | 0.597 | 0.715 | 0.834 | 0.880 |

### 🏆 Final Model Selection Rationale
While a single Decision Tree scored a slightly higher raw F1 (0.813 vs. 0.801), individual tree leaves output polarized 0/1 probabilities. For operational triage, **Extra Trees** was selected because it achieved the highest ranking discrimination (**0.891 ROC-AUC**, **0.916 PR-AUC**) and well-calibrated probabilities.

![Model Comparison](images/Model_Comparison.png)

---

## 📈 Model Evaluation & Risk Tiers

Test set performance metrics and operational confusion matrix:

![Metric_Scores](images/Metric_Scores.png)
![Confusion_Matrix](images/Confusion_Matrix.png)
![ROC_Curve](images/ROC_Curve.png)

### Operational Risk Triage Strategy
Continuous probabilities from the test set (36,104 orders) were bucketed into operational dispatch tiers:

| Risk Tier | Probability Range | Orders (n) | Actual Late Rate | Recommended Action |
| :--- | :---: | :---: | :---: | :--- |
| **Low Risk** | p < 0.30 | 11,520 (31.9%) | **5.7%** | Standard automated processing |
| **Medium Risk** | 0.30 <= p < 0.60 | 10,063 (27.9%) | **57.8%** | Watchlist; prioritize if warehouse load spikes |
| **High Risk** | p >= 0.60 | 14,521 (40.2%) | **91.8%** | Immediate priority picking & carrier reassignment |

*Tuning the operational decision threshold from 0.50 down to 0.40 lifts recall to **87.7%**, catching over 2,400 additional late deliveries.*

---

## 🔍 Feature Importance & Explainability

![Feature_Importance](images/Feature_Importance.png)

- **Primary Predictive Drivers:** `Shipping Mode (Standard Class)`, `Urgent_Shipment` (scheduled transit <= 2 days), and scheduled transit days dictate model decisions.
- **SHAP Insights:** Standard Class shipping consistently exerts positive attribution toward delay risk, whereas larger scheduled transit buffers reduce delay probability.

---

## 🎯 Error Analysis: Where The Model Struggles

Breaking down model performance across shipping modes on the test set revealed an important operational insight:

| Shipping Mode | Test Samples (n) | Precision | Recall | F1 Score | Operational Insight |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **First Class** | 5,628 | 1.000 | 1.000 | 1.000 | Strict 1-day SLA; virtually deterministic |
| **Same Day** | 1,975 | 0.983 | 0.983 | 0.983 | Dedicated express routing; rarely misses |
| **Second Class** | 7,005 | 0.834 | 0.991 | 0.905 | High delay rate, but readily caught |
| **Standard Class** | 21,496 | 0.708 | 0.403 | **0.513** | **Responsible for >95% of all false negatives** |

**Why Standard Class Fails:** Standard Class features a wide 4 to 6 day scheduled window. Delays in this tier depend heavily on in-transit disruptions (hub congestion, courier delays, weather) that are unobservable at order checkout.

---

## 🖥️ Streamlit Interactive Dashboard

An interactive dashboard is available in `app/app.py`:
- **Executive Summary:** Real-time test set metric cards, confusion matrix, and risk tier breakdowns.
- **Delay Heatmaps:** Delay rates across global destination markets and shipping methods.
- **Live Risk Predictor:** Enter order details to receive an instant calibrated probability, Low/Medium/High risk badge, and natural language risk explanations.

```bash
streamlit run app/app.py
```

---

## 📁 Repository Structure

```
├── app/
│   └── app.py                     # Streamlit interactive dashboard
├── data/
│   ├── raw/                       # Raw DataCo dataset
│   └── processed/                 # Cleaned splits & model comparison benchmarks
├── images/                        # Plots, metrics, confusion matrices, ROC curves
├── models/
│   └── best_model.pkl             # Trained Extra Trees production pipeline
├── notebooks/                     # 10 sequential analysis notebooks (01 to 10)
├── sql/                           # 24 analytical SQL queries across 3 domains
├── src/                           # Modular Python pipeline modules
├── requirements.txt               # Project dependencies
└── README.md
```

---

## 🛠️ Tech Stack

- **Core & Data Manipulation:** Python, Pandas, NumPy, SQLite
- **Machine Learning:** Scikit-learn (Pipelines, ExtraTrees, HistGradientBoosting)
- **Model Explainability & Visualization:** SHAP, Matplotlib, Seaborn
- **Statistical Testing & Time Series:** SciPy (Stats), Statsmodels (Holt-Winters)
- **Web Dashboard:** Streamlit

---

## ⚙️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/YogeshKhillare04/supply-chain-risk-intelligence.git
cd supply-chain-risk-intelligence

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place DataCoSupplyChainDataset.csv inside data/raw/ (see data/README.md)

# 4. Launch the Streamlit web dashboard
streamlit run app/app.py

# 5. Run Jupyter notebooks in order (01 to 10)
jupyter notebook notebooks/
```

---

## ⚠️ Limitations & Future Work

- **Observational Data:** Relies on historical tabular data; does not include live GPS telemetry or dynamic transit weather feeds.
- **Standard Class Transit Visibility:** Order-time features reach an information ceiling for long-haul standard shipments without in-transit checkpoints.
- **Simulation vs Production:** The A/B test is a counterfactual statistical simulation rather than an active in-production trial.

---

⭐ If you found this project insightful, consider giving it a star!
