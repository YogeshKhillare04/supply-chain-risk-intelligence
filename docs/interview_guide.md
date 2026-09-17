# Interview Preparation Guide

This document covers the key technical concepts behind each major component of the project. For every major decision, you should be able to explain: **why**, **how**, **alternatives**, and **limitations**.

---

## 1. Data Leakage

**Q: What is data leakage and how did you handle it?**

Data leakage happens when information from outside the training dataset is used to create the model, giving it unrealistic advantages.

In this project, three columns were leakage:
- `Days for shipping (real)` — the actual shipping time, only known AFTER delivery
- `Delivery Status` — the final outcome, which IS what we're predicting
- `shipping date (DateOrders)` — when it actually shipped, not known at order time

I also fixed a subtler leak: aggregated features (like average sales per category) were originally computed on the full dataset before train/test split. This means the model was indirectly seeing test data during training. I moved these computations to be train-only.

**Key concept:** For any feature, ask: "Would I know this at prediction time?"

---

## 2. Train/Test Split Strategy

**Q: Why stratified split? Why not random?**

The target is mildly imbalanced (55% late, 45% on-time). Stratified splitting ensures both train and test sets have the same class distribution. Without it, random variation could give us a test set with a very different balance, making evaluation unreliable.

**Q: Why split before feature engineering?**

Aggregated features (like category average sales) computed on the full dataset would leak information. If category X has 100 orders (80 train, 20 test), computing the average on all 100 means the training features contain information from test orders.

---

## 3. Model Selection

**Q: Why did you start with a DummyClassifier?**

A baseline establishes the minimum performance. If a complex model can't beat "always predict the majority class," something is wrong. The Dummy classifier uses the most frequent class strategy.

**Q: Why tree-based models?**

The statistical analysis showed weak linear correlations between features and the target. This means the relationship is likely non-linear. Tree-based models (Random Forest, Extra Trees, Gradient Boosting) can capture non-linear interactions between features, which linear models like Logistic Regression cannot.

**Q: What's the difference between Random Forest and Extra Trees?**

Both are ensemble methods that build many decision trees. The key difference:
- **Random Forest**: finds the best split at each node (from a random subset of features)
- **Extra Trees**: uses random splits (not optimized) — this adds more randomness, which can reduce overfitting and is faster to train

**Q: Why not deep learning?**

This is tabular data with ~180K rows. Tree-based ensembles consistently outperform neural networks on tabular data (as shown by benchmarks). Deep learning would add complexity without benefit and would be harder to explain.

---

## 4. Evaluation Metrics

**Q: Why not just use accuracy?**

With 55% late deliveries, a model that always predicts "late" gets 55% accuracy — not useful. We need:

- **Precision**: "When the model says 'late', how often is it right?" (reduces false alarms)
- **Recall**: "Of all actual late deliveries, how many did we catch?" (reduces missed delays)
- **F1**: Harmonic mean of precision and recall — balances both
- **ROC-AUC**: How well the model discriminates between classes overall
- **PR-AUC**: Particularly useful when we care about the positive class (late deliveries)

**Q: Which metric matters most for this problem?**

It depends on the business context:
- If intervening on flagged orders is cheap → optimize for **recall** (catch more)
- If interventions are expensive → optimize for **precision** (fewer false alarms)
- F1 is a reasonable default balance

---

## 5. Threshold Selection

**Q: Why is 0.5 not always the best threshold?**

The default 0.5 threshold assumes equal costs for false positives and false negatives. In delivery risk:
- Missing a late delivery (false negative) might cost customer satisfaction
- Unnecessarily intervening (false positive) costs operational resources

If missing delays is more costly, lower the threshold (e.g., 0.3) to catch more at the cost of more false alarms.

The threshold analysis in notebook 06 shows precision/recall at multiple thresholds.

---

## 6. Risk Categories

**Q: How did you choose the risk thresholds (0.3, 0.7)?**

These are reasonable defaults, not optimized values. I chose them based on:
- Below 0.3: model is fairly confident about on-time delivery
- 0.3–0.7: uncertain zone
- Above 0.7: model is fairly confident delivery will be late

In production, you'd tune these based on operational capacity and business priorities.

---

## 7. Error Analysis

**Q: What did you learn from analyzing model errors?**

Error analysis reveals systematic patterns in mistakes:
- Are certain shipping modes harder to predict?
- Does the model perform worse for specific markets?
- Are false negatives concentrated in borderline probability ranges?

This informs whether to:
- Use different thresholds for different segments
- Collect more data for underperforming areas
- Add segment-specific features

---

## 8. SHAP Explainability

**Q: What is SHAP?**

SHAP (SHapley Additive exPlanations) is based on game theory (Shapley values). It assigns each feature a contribution to the prediction, explaining HOW MUCH each feature pushed the prediction toward "late" or "on-time."

**Q: What's the difference between SHAP and tree-based feature importance?**

- **Tree importance**: measures how much each feature reduces impurity across all trees. It tells you WHICH features matter but not the DIRECTION.
- **SHAP**: tells you both which features matter AND whether their high/low values push toward late or on-time. It's also consistent (sum of SHAP values = prediction).

**Q: Can you make causal claims from SHAP?**

No. SHAP shows correlation/association in the model, not causation. "Shipping Mode contributed to the high risk prediction" ≠ "Shipping Mode caused the delay."

---

## 9. Statistical Tests

**Q: Why chi-square for shipping mode vs late delivery?**

Both variables are categorical (shipping mode: 4 categories, late delivery: binary). Chi-square tests whether two categorical variables are independent.

**Q: Why Welch's t-test instead of regular t-test?**

Welch's t-test doesn't assume equal variances between groups. Since late and on-time deliveries may have different variance in scheduled days, Welch's is more robust.

**Q: What does "statistically significant" mean?**

A p-value < 0.05 means there's less than 5% probability of seeing this result if there were truly no relationship. It does NOT mean the effect is practically important — with 180K rows, even tiny effects can be "significant."

---

## 10. Forecasting

**Q: Why simple methods instead of LSTM?**

- The dataset has ~3 years of weekly data = ~150 data points. That's too little for deep learning.
- Moving averages and exponential smoothing work well for short-term forecasting
- They're interpretable and fast
- The goal is demonstrating the methodology, not achieving state-of-the-art accuracy

**Q: How is forecasting relevant to delivery risk?**

If we forecast high order volume for next week AND the risk model identifies many orders as high-risk, we can proactively allocate more delivery resources. This is exactly what food delivery companies do for lunch/dinner rush predictions.

---

## 11. A/B Testing

**Q: Why is this a simulation?**

We don't have real experimental data. In practice, a company would:
1. Randomly assign orders to control (no intervention) vs treatment (model-based intervention)
2. Measure actual delivery outcomes
3. Test if the difference is statistically significant

Our simulation demonstrates the methodology with synthetic intervention effects.

**Q: What's the difference between statistical significance and practical significance?**

A result can be statistically significant (p < 0.05) but practically meaningless (0.1% improvement). In A/B testing, you need to define a Minimum Detectable Effect upfront — how big does the improvement need to be to justify the cost?

---

## 12. SQL Queries

**Q: Why SQL alongside Python?**

In industry, data scientists use SQL for:
- Exploring data before loading into Python
- Quick ad-hoc analysis
- Production queries and dashboards
- Working with data too large for memory

The SQL queries answer the same business questions as the EDA but demonstrate proficiency in a different tool.

**Q: Can you explain a window function you used?**

`RANK() OVER (PARTITION BY Market ORDER BY AVG(Late_delivery_risk) DESC)` ranks shipping modes by their late delivery rate WITHIN each market separately. This lets us find the highest-risk shipping mode for each market.

---

## 13. General Data Science Questions

**Q: If you had more time, what would you improve?**

1. Hyperparameter tuning (GridSearch/RandomizedSearch)
2. More sophisticated feature engineering (interaction features)
3. Deploy as an API for real-time predictions
4. Add monitoring for model drift
5. Collect additional data on external factors (weather, holidays)

**Q: How would this approach apply to food delivery (Swiggy)?**

The same methodology transfers directly:
- **Target**: Predict if an order will be delivered late
- **Features**: Restaurant preparation time, rider availability, distance, traffic, weather, time of day
- **Risk categories**: Prioritize which delayed orders to proactively address
- **A/B testing**: Test if model-based rider assignment improves delivery times
- **Forecasting**: Predict demand by zone and time to optimize rider positioning

**Q: What are the limitations of this project?**

1. The DataCo dataset is not real-time — patterns may not reflect live operations
2. The model doesn't account for external factors (weather, events, traffic)
3. Aggregated features computed from training data may become stale over time
4. The A/B test is simulated, not a real experiment
5. The forecasting uses coarse weekly granularity
