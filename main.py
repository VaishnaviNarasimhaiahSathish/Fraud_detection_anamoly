from src.data_preprocessing import load_data, prepare_data, split_data, apply_smote
from src.eda import plot_class_distribution, plot_amount_stats, plot_correlation, scatter_plot
from src.model import train_random_forest
from src.evaluation import evaluate_model, plot_precision_recall
from src.visualization import plot_feature_importances

def main():
    # Step 1: Load data
    df = load_data("data/creditcard.csv")
    print("Data loaded:", df.shape)

    # Step 2: EDA
    plot_class_distribution(df)
    plot_amount_stats(df)
    corr = plot_correlation(df)
    scatter_plot(df, "V2", "V4")  # Example scatter plot

    # Step 3: Preprocessing
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    # Step 4: Train Model
    rf_model = train_random_forest(X_train_res, y_train_res)

    # Step 5: Evaluate Model
    y_pred = evaluate_model(rf_model, X_test, y_test)

    # Step 6: Precision-Recall Curve
    plot_precision_recall(rf_model, X_test, y_test)

    # Step 7: Feature Importances
    plot_feature_importances(rf_model, X.columns)

if __name__ == "__main__":
    main()
