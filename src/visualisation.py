import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Plot model performance comparison
def plot_model_comparison(results_dict):
    df = pd.DataFrame(results_dict).T
    figs = []

    for metric in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=df.index, y=df[metric], ax=ax, palette="crest")
        ax.set_title(f"{metric} Comparison")
        ax.set_ylabel(metric)
        ax.set_xlabel("Models")
        ax.set_ylim(0, 1)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')

        figs.append(fig)

    return figs

# Plot confusion matrices
def plot_confusion_matrices(predictions_dict):
    figs = []
    for model_name, (y_true, y_pred) in predictions_dict.items():
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"Confusion Matrix - {model_name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        figs.append(fig)

    return figs
