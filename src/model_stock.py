import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def train_model(
    df: pd.DataFrame,
    model_path="models/random_forest.pkl",
    output_csv="outputs/predictions.csv",
    verbose: bool = True,
):
    """
    Train a RandomForestClassifier to predict next-day price movement.

    Args:
        df (pd.DataFrame): Feature-enhanced stock data
        model_path (str): Path to save the model
        output_csv (str): Path to save predictions
        verbose (bool): Whether to print metrics and plot feature importance

    Returns:
        model (RandomForestClassifier): Trained model
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    features = ['Return', 'SMA_5', 'SMA_10', 'Volatility', 'RSI']
    df = df.dropna(subset=features + ['Target'])

    X = df[features]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]  # Probability of "Up"

    # Save model
    joblib.dump(model, model_path)

    # Add predictions to dataframe
    df_test = df.iloc[-len(y_test):].copy()
    df_test['Predicted'] = preds
    df_test['Confidence'] = probs
    df_test.to_csv(output_csv, index=False)

    if verbose:
        print("\nModel Evaluation:")
        print("Accuracy:", round(accuracy_score(y_test, preds), 3))
        print(classification_report(y_test, preds, target_names=["Down", "Up"]))
        print(f"Latest prediction confidence: {round(probs[-1], 3)}")
        plot_feature_importance(model, features)

    return model


def load_model(model_path="models/random_forest.pkl"):
    """
    Load a saved RandomForest model from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)


def plot_feature_importance(model, features):
    importances = model.feature_importances_
    indices = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)

    plt.figure(figsize=(8, 4))
    plt.title("Feature Importance")
    plt.bar([features[i] for i in indices], [importances[i] for i in indices])
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()
