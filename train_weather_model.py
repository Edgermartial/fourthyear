import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# === Path to dataset ===
DATA_PATH = r"C:\Users\edger\Desktop\model\myproject\seattle-weather.csv"

# === Output directory (inside myapp/) ===
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "myapp")
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_PATH = os.path.join(OUTPUT_DIR, "weather_model.pkl")

# === Load dataset ===
df = pd.read_csv(DATA_PATH)

# Ensure required columns exist
required_cols = ["precipitation", "temp_max", "temp_min", "wind", "date"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# === Preprocessing ===
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# Rolling features
df["precip_7d_mean"] = df["precipitation"].rolling(7).mean().shift(1)
df["tempmax_7d_mean"] = df["temp_max"].rolling(7).mean().shift(1)
df["tempmin_7d_mean"] = df["temp_min"].rolling(7).mean().shift(1)
df["wind_7d_mean"] = df["wind"].rolling(7).mean().shift(1)
df["precip_3d_sum"] = df["precipitation"].rolling(3).sum().shift(1)

# Calendar features
df["month"] = df["date"].dt.month
df["day_of_year"] = df["date"].dt.dayofyear

# Target: 1 = suitable, 0 = unsuitable
df["target"] = (
    (df["temp_min"] >= 5) &
    (df["temp_max"] <= 30) &
    (df["precipitation"] < 10)
).astype(int)

# Predict next day suitability
df["target_nextday"] = df["target"].shift(-1)

# Drop rows with NaNs (from rolling)
df = df.dropna()

# === Train/test split (time-aware: first 80% train, last 20% test) ===
n_train = int(len(df) * 0.8)
train_df = df.iloc[:n_train]
test_df = df.iloc[n_train:]

features = [
    "precip_7d_mean", "tempmax_7d_mean", "tempmin_7d_mean",
    "wind_7d_mean", "precip_3d_sum", "month", "day_of_year"
]

X_train, y_train = train_df[features], train_df["target_nextday"]
X_test, y_test = test_df[features], test_df["target_nextday"]

# === Train model ===
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# === Save model ===
joblib.dump(model, MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
