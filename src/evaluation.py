"""
Model evaluation utilities.

Functions for evaluating classification models, creating risk categories
from probabilities, and generating evaluation plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a classification model with multiple metrics.
    
    Parameters
    ----------
    model : fitted sklearn estimator
        Must support predict() and predict_proba()
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        True labels
    model_name : str
        Name for display purposes
    
    Returns
    -------
    dict
        Dictionary of metric name → score
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_prob),
        "PR_AUC": average_precision_score(y_test, y_prob),
    }
    
    return metrics


def get_risk_category(probability, low_threshold=0.3, high_threshold=0.7):
    """
    Convert a late-delivery probability into a risk category.
    
    Thresholds are set based on practical considerations:
    - Below 0.3: model is fairly confident delivery will be on time
    - 0.3 to 0.7: uncertain zone — could go either way
    - Above 0.7: model is fairly confident delivery will be late
    
    These thresholds are NOT "optimal" — they are reasonable defaults.
    The notebook (06_model_evaluation) explores different thresholds.
    
    Parameters
    ----------
    probability : float or array-like
        Probability of late delivery (from predict_proba)
    low_threshold : float
        Below this = Low Risk (default 0.3)
    high_threshold : float
        Above this = High Risk (default 0.7)
    
    Returns
    -------
    str or array-like
        Risk category: "Low Risk", "Medium Risk", or "High Risk"
    """
    if isinstance(probability, (int, float)):
        if probability < low_threshold:
            return "Low Risk"
        elif probability < high_threshold:
            return "Medium Risk"
        else:
            return "High Risk"
    
    # Handle arrays
    probability = np.asarray(probability)
    categories = np.where(
        probability < low_threshold, "Low Risk",
        np.where(probability < high_threshold, "Medium Risk", "High Risk")
    )
    return categories


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Plot a confusion matrix with labeled axes.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    title : str
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["On Time", "Late"],
        yticklabels=["On Time", "Late"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_model_comparison(results_df):
    """
    Plot a grouped bar chart comparing models across metrics.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Must have a "Model" column and metric columns
    """
    metrics = [c for c in results_df.columns if c != "Model"]
    melted = results_df.melt(id_vars="Model", value_vars=metrics,
                              var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=melted, x="Metric", y="Score", hue="Model")
    plt.ylim(0, 1.05)
    plt.title("Model Comparison")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def threshold_analysis(y_true, y_prob, thresholds=None):
    """
    Show precision, recall, and F1 at different classification thresholds.
    
    This helps understand the trade-off between catching late deliveries
    (recall) and avoiding false alarms (precision).
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_prob : array-like
        Predicted probabilities for positive class
    thresholds : list of float, optional
        Thresholds to evaluate (default: 0.3 to 0.7)
    
    Returns
    -------
    pd.DataFrame
        Metrics at each threshold
    """
    if thresholds is None:
        thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    
    results = []
    for t in thresholds:
        y_pred = (np.asarray(y_prob) >= t).astype(int)
        results.append({
            "Threshold": t,
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "Predicted_Positive_Rate": y_pred.mean(),
        })
    
    return pd.DataFrame(results)
