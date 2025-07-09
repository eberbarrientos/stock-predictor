import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def train_model(df: pd.DataFrame, model_path="models/random_forest.pkl", verbose: bool = True):
    """
    Train a RandomForestClassifier to predict next-day price movement.

    Args:
        df (pd.DataFrame): Feature-enhanced stock data
        verbose (bool): Whether to print metrics

    Returns:
        model (RandomForestClassifier): Trained model
    """

    # Target: 1 if tomorrow's close > today's, else 0
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    features = ['Return', 'SMA_5', 'SMA_10', 'Volatility', 'RSI']
    df = df.dropna(subset=features + ['Target'])  # drop rows with NaNs

    X = df[features]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]  # probability of "Up"

    latest_conf = probs[-1]  # confidence for the most recent prediction

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, model_path)

    if verbose:
        print("\nModel Evaluation:")
        print("Accuracy:", round(accuracy_score(y_test, preds), 3))
        print(classification_report(y_test, preds, target_names=["Down", "Up"]))
        print(f"Latest prediction confidence: {round(latest_conf, 3)}")


    return model, latest_conf

def load_model(model_path="models/random_forest.pkl"):
    """
    Load a saved RandomForest model from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)

