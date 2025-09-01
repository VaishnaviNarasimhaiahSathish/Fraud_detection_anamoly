**CREDIT CARD FRAUD DETECTION**

**Overview:**

This project explores fraud detection using machine learning techniques on a highly imbalanced dataset (credit card transactions). The goal is to identify fraudulent transactions while minimizing false positives.

**Key highlights:**

1. Exploratory Data Analysis (EDA) and visualization of transaction patterns

2. Supervised learning using Random Forest

3. Handling class imbalance with SMOTE (Synthetic Minority Oversampling)

4. Model evaluation using precision, recall, F1-score, confusion matrix, PR curves

5. Modularized pipeline (src/ folder) for clean and reproducible code

**Project Structure:**
fraud_detection_anomaly/

1. notebooks/                   (_Jupyter notebooks for experiments_
    1. fraud_detection.ipynb     (_Original notebook_)

2. results/                     (_Outputs (images, plots, reports)_)


3. src/                         (_Source code_)
    1. data_preprocessing.py    (_Data loading, cleaning, scaling_)
    2. eda.py                   (_Exploratory Data Analysis (correlations, plots)_)
    3. model.py                 (_ML models (Random Forest, Isolation Forest, etc.)_)
    4. evaluation.py            (_Evaluation metrics, confusion matrix, PR curve_)
    5. visualization.py         (_Visualization functions (feature importance, etc.)_)


4. main.py                      (_ Main script to run the pipeline_)

5. requirements.txt             (_Project dependencies_)
6. .gitignore                   (_Files/folders to ignore_)
7. README.md                    (_Project documentation_)


**Installation:**

1. Clone the repository:

    git clone https://github.com/VaishnaviNarasimhaiahSathish/Fraud_detection_anamoly.git

    cd Fraud_detection_anamoly

2. Create a virtual environment and install dependencies:

    python3 -m venv venv
    source venv/bin/activate    (_(Mac/Linux)_)
    venv\Scripts\activate       (_(Windows)_)

    pip install -r requirements.txt

**Usage:**

Run the full pipeline:
python main.py

**Results:**

1. Without balancing: model struggled with recall (missing fraud cases).
2. With SMOTE + Random Forest:
3. Precision (fraud class): ~0.83
4. Recall (fraud class): ~0.83
5. Overall accuracy: ~99.9%

The model successfully reduced false negatives, which is crucial for fraud detection.

**Future Improvements:**

1. Hyperparameter tuning for Random Forest (RandomizedSearchCV)
2. Compare with unsupervised anomaly detection (Isolation Forest, Autoencoders)
3. Deploy as an API (FastAPI / Flask) or interactive dashboard (Streamlit)

**Dataset:**

The dataset used is the Kaggle Credit Card Fraud Detection dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

**Author:**

Vaishnavi Narasimhaiah Sathish
Master’s Student in Computer Science | Germany
