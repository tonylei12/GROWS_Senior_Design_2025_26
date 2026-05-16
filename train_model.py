import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# Features must be computable at inference time from Open-Meteo's live API
FEATURE_COLS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
    "rain_last_24h",
    "rain_last_3d",
    "rain_last_7d",
    "days_since_rain",
    "et0_last_7d",
]

df = pd.read_csv("training_data.csv")
df = df.dropna(subset=FEATURE_COLS + ["label"])
print(f"Training on {len(df)} samples")
print(f"Class balance: {df['label'].value_counts().to_dict()}")

X = df[FEATURE_COLS]
y = df["label"]

# Time-aware split: train on older data, test on newer data
# This is more honest than random splitting for time-series-like data
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

model = RandomForestClassifier(
    n_estimators=200, max_depth=12, random_state=42,
    class_weight="balanced", n_jobs=-1,
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("\n=== Performance on held-out recent data ===")
print(classification_report(y_test, preds))
print("Confusion matrix [[TN FP] [FN TP]]:")
print(confusion_matrix(y_test, preds))

# Cross-validation for a more honest estimate
cv = cross_val_score(model, X, y, cv=5, scoring="f1", n_jobs=-1)
print(f"\n5-fold CV F1: {cv.mean():.3f} (+/- {cv.std()*2:.3f})")

print("\nFeature importance:")
for f, imp in sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {f}: {imp:.3f}")

joblib.dump({
    "model": model,
    "feature_cols": FEATURE_COLS,
}, "water_model.pkl")
print("\nSaved water_model.pkl")

