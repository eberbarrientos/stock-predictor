import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

def train_etf_model(df: pd.DataFrame, model_path="models/etf_model.pkl", horizon_days=365, verbose=True):
    target_col = f'Target_{horizon_days}d'
    df[target_col] = (df['Close'].shift(-horizon_days) > df['Close']).astype(int)

    features = ['Return', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200', 'Volatility', 'RSI']
    df = df.dropna(subset=features + [target_col])
    X, y = df[features], df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_res, y_train_res)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, model_path)

    if verbose:
        print(f"\n[ETF Model Evaluation - {horizon_days}-day forward with SMOTE]")
        print("Accuracy:", round(accuracy_score(y_test, preds), 3))
        print(classification_report(y_test, preds, target_names=["Down", "Up"]))
        print(f"Latest prediction confidence: {round(probs[-1], 3)}")
        plot_feature_importance(model, features)

    return model

def load_etf_model(model_path="models/etf_model_30d.pkl"):
    """
    Load a saved ETF model from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)

def plot_feature_importance(model, features):
    importances = model.feature_importances_
    indices = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)

    plt.figure(figsize=(8, 4))
    plt.title("ETF Model - Feature Importance")
    plt.bar([features[i] for i in indices], [importances[i] for i in indices])
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()
