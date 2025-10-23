# crop_rec.py
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# ================= Step 1: Load dataset =================
file_path = r"C:\Users\edger\Desktop\model\myproject\Crop_Recommendation.csv"
df = pd.read_csv(file_path)

print("✅ Dataset loaded successfully!")
print("Columns in dataset:", df.columns.tolist())

# Features and target
X = df.drop("Crop", axis=1)
y = df["Crop"]

# Encode crop labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ================= Step 2: Train-test split =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ================= Step 3: Define model + hyperparameter tuning =================
rf = RandomForestClassifier(random_state=42)

param_dist = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False]
}

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=20,              # try 20 random combos
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

print("\n🚀 Running hyperparameter tuning...")
random_search.fit(X_train, y_train)

best_rf = random_search.best_estimator_
print("\n✅ Best parameters found:", random_search.best_params_)

# ================= Step 4: Evaluate =================
y_pred = best_rf.predict(X_test)
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# ================= Step 5: Save model & encoder =================
# Save inside myapp/ so Django can access
save_dir = r"C:\Users\edger\Desktop\model\myproject\myapp"
os.makedirs(save_dir, exist_ok=True)

joblib.dump(best_rf, os.path.join(save_dir, "crop_model.pkl"))
joblib.dump(le, os.path.join(save_dir, "label_encoder.pkl"))

print(f"\n🎉 Model and LabelEncoder saved to: {save_dir}")
