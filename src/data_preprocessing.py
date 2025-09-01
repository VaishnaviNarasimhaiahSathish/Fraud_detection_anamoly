import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_data(path: str):
    """Load the dataset from CSV."""
    df = pd.read_csv(path)
    return df

def prepare_data(df):
    """Split features and target."""
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split into train/test sets with stratification."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def apply_smote(X_train, y_train, random_state=42):
    """Balance the dataset using SMOTE."""
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res
