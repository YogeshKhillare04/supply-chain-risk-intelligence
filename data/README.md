# Dataset Setup

This project uses the **DataCo Smart Supply Chain Dataset** from Kaggle.

## Download Instructions

1. Go to: https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis
2. Download the dataset (you'll need a free Kaggle account)
3. Extract the ZIP file
4. Place `DataCoSupplyChainDataset.csv` into `data/raw/`

## Repository File Structure

```
data/
├── raw/
│   └── DataCoSupplyChainDataset.csv    ← Download from Kaggle (~91 MB, not in repo)
├── processed/
│   ├── model_comparison.csv            ← ✅ Included — 6-model benchmark results
│   ├── y_test.csv                      ← ✅ Included — test set labels (36,104 rows)
│   ├── y_train.csv                     ← ✅ Included — train set labels (144,415 rows)
│   ├── cleaned_data.csv                ← ❌ Not in repo (~55 MB) — regenerate via notebook 02
│   ├── cleaned_with_dates.csv          ← ❌ Not in repo (~59 MB) — regenerate via notebook 02
│   ├── X_train.csv                     ← ❌ Not in repo (~57 MB) — regenerate via notebook 03
│   └── X_test.csv                      ← ❌ Not in repo (~14 MB) — regenerate via notebook 03
└── README.md
```

## Dataset Overview

- **Records**: 180,519 supply chain transactions
- **Features**: 53 columns covering orders, shipping, customers, products, and financials
- **Target Variable**: `Late_delivery_risk` (binary: 1 = late, 0 = on time)
- **Class Balance**: 54.8% late (1) / 45.2% on time (0)
- **Train / Test Split**: 144,415 train / 36,104 test (stratified 80/20)
- **Time Range**: 2015–2018
- **Source**: DataCo Global (anonymized supply chain data)

## Key Columns Used

| Column | Type | Description |
|--------|------|-------------|
| `Late_delivery_risk` | Target | Whether delivery was late (1) or on time (0) |
| `Days for shipment (scheduled)` | Feature | Planned shipping duration in days |
| `Days for shipping (real)` | ⚠️ Removed | Actual shipping duration — **data leakage** |
| `Delivery Status` | ⚠️ Removed | Final delivery outcome — **data leakage** |
| `Shipping Mode` | Feature | First Class, Second Class, Standard Class, Same Day |
| `Market` | Feature | Geographic market (Africa, Europe, LATAM, Pacific Asia, USCA) |
| `Order Region` | Feature | Specific region within market |
| `Category Name` | Feature | Product category |
| `Customer Segment` | Feature | Consumer, Corporate, Home Office |
| `Sales` | Feature | Order sales value |
| `Order Item Quantity` | Feature | Number of items ordered |
| `Benefit per order` | Feature | Profit/benefit from the order |

## Regenerating Processed Files

Run the notebooks in order to regenerate all processed data from scratch:

```bash
# After placing DataCoSupplyChainDataset.csv in data/raw/
jupyter notebook notebooks/

# Run in order:
# 01_data_exploration.ipynb
# 02_data_preprocessing.ipynb   → produces cleaned_data.csv, cleaned_with_dates.csv
# 03_feature_engineering.ipynb  → produces X_train.csv, X_test.csv, y_train.csv, y_test.csv
# 05_model_training.ipynb       → produces model_comparison.csv, best_model.pkl
```

## Why Large Files Are Excluded

The raw dataset (~91 MB) and large processed CSVs (~185 MB total) exceed GitHub's recommended
per-push limits and are excluded via `.gitignore`. The small benchmark outputs (`model_comparison.csv`,
`y_test.csv`, `y_train.csv`) are included since they are small and useful as reference without
needing to re-run training.
