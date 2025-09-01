import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances(model, feature_names, top_n=10):
    """Plot top N important features from model."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title("Top Feature Importances")
    plt.show()
