from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train, class_weight="balanced", random_state=42):
    """Train Random Forest classifier."""
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return rf
