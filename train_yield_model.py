import os
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# === Path to dataset ===
DATA_PATH = r"C:\Users\edger\Desktop\myproject\climate-ds.csv"

# === Output directory (inside Django app folder, e.g., myapp/) ===
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "myapp")
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_PATH = os.path.join(OUTPUT_DIR, "yield_model.pkl")

# === Load dataset ===
print(f"📂 Loading dataset from {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# === Identify target column ===
# Automatically find the "yield" column
possible_targets = [c for c in df.columns if 'yield' in c.lower()]
if not possible_targets:
    raise ValueError("❌ Could not find any 'yield' column in dataset. Please rename your target column (e.g. 'Yield').")

TARGET_COL = possible_targets[0]
print(f"🎯 Using target column: {TARGET_COL}")

# === Drop unwanted columns ===
drop_cols = [col for col in ['Unnamed: 0', 'Year'] if col in df.columns]
for col in drop_cols:
    df = df.drop(columns=[col])
    print(f"🗑️ Dropped column: {col}")

# === Prepare features and target ===
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Drop missing targets
mask = y.notnull()
X, y = X[mask], y[mask]

# === Identify feature types ===
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"📊 Numeric features: {numeric_features}")
print(f"📋 Categorical features: {categorical_features}")

# === Clean numeric columns (convert to float safely) ===
for col in numeric_features:
    X[col] = pd.to_numeric(X[col], errors='coerce')

# === Split data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Preprocessing pipeline ===
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# === Define model ===
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

# === Full pipeline ===
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# === Train model ===
print("🚀 Training model ...")
pipeline.fit(X_train, y_train)

# === Evaluate ===
print("📊 Evaluating model ...")
y_pred = pipeline.predict(X_test)

# For compatibility with older scikit-learn
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=== Evaluation Metrics ===")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# === Save model ===
joblib.dump(pipeline, MODEL_PATH)
print(f"\n✅ Model saved to {MODEL_PATH}")

# === Save metadata ===
metadata = {
    "target": TARGET_COL,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "n_rows": len(df),
    "n_features": len(X.columns)
}
META_PATH = os.path.join(OUTPUT_DIR, "yield_metadata.pkl")
joblib.dump(metadata, META_PATH)
print(f"🧠 Metadata saved to {META_PATH}")

# === Optional: Show top 10 feature importances ===
try:
    importances = pipeline.named_steps["model"].feature_importances_
    print("\n🔥 Top 10 Important Features:")
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(cols)
        elif name == 'cat':
            onehot = transformer.named_steps['onehot']
            feature_names.extend(onehot.get_feature_names_out(cols))
    sorted_idx = np.argsort(importances)[::-1]
    for idx in sorted_idx[:10]:
        print(f"  {feature_names[idx]}: {importances[idx]:.4f}")
except Exception as e:
    print(f"(Feature importance skipped: {e})")
