import matplotlib.pyplot as plt
import seaborn as sns

def plot_class_distribution(df):
    """Plot fraud vs non-fraud class distribution."""
    sns.countplot(x="Class", data=df)
    plt.title("Class Distribution")
    plt.show()

def plot_amount_stats(df):
    """Show average transaction amount per class."""
    print(df.groupby("Class")["Amount"].mean())

def plot_correlation(df):
    """Show correlation of features with fraud class."""
    corr = df.corr()["Class"].sort_values(ascending=False)
    print(corr.head(10))
    return corr

def scatter_plot(df, col_x, col_y):
    """Scatter plot of two variables colored by class."""
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=df[col_x], y=df[col_y], hue=df["Class"], alpha=0.5, palette="coolwarm")
    plt.title(f"{col_x} vs {col_y}")
    plt.show()
