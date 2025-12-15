import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

# --- CONFIGURATION ---
INPUT_FILE = "data/model_ready_data.csv"
MODEL_FILE = "data/wildlife_model.pkl"
PLOT_FILE = "assets/feature_importance.png"

# Ensure assets folder exists
os.makedirs("assets", exist_ok=True)

# --- 1. LOAD & CLEAN DATA ---
print("📂 Loading Dataset...")
try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"❌ Error: {INPUT_FILE} not found.")
    exit()

print(f"   Raw Data: {len(df)} rows")

# DEFINE FEATURES
feature_cols = ['elevation', 'dist_water', 'dist_forest', 'dist_village']

# --- BULLETPROOF CLEANING (Fixes the NaN Error) ---
# 1. Convert columns to numbers (forces any text errors to NaN)
for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 2. Replace "0" and "-1" (API failures) with NaN so we can treat them as missing
df[feature_cols] = df[feature_cols].replace({0: np.nan, -1: np.nan})

# 3. Fill Missing Values (The Critical Step)
# Strategy: Use the "Mean" (Average) of the column to fill gaps.
# If the whole column is empty (worst case), fill with 0.
imputer = SimpleImputer(strategy='mean')

# Check if we have enough valid data to impute
if df[feature_cols].isnull().all().all():
    print("❌ CRITICAL ERROR: Your feature columns are COMPLETELY empty.")
    print("   The extraction script failed to get ANY data.")
    print("   Solution: Run 'extract_features_robust.py' again and let it finish.")
    exit()

try:
    df[feature_cols] = imputer.fit_transform(df[feature_cols])
except ValueError:
    # Fallback if imputation fails
    print("⚠️ Imputation warning: Filling remaining gaps with 0.")
    df[feature_cols] = df[feature_cols].fillna(0)

# Final check for NaNs
if df[feature_cols].isnull().any().any():
    df[feature_cols] = df[feature_cols].fillna(0)

# DEFINE X and y
X = df[feature_cols]
y = df['Target']  # 1 = Conflict, 0 = Safe

# Split: 80% for Training, 20% for Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"🧠 Training Model on {len(X_train)} rows...")
print(f"🧪 Testing Model on {len(X_test)} rows...")

# --- 2. TRAIN RANDOM FOREST MODEL ---
model = RandomForestClassifier(
    n_estimators=100,      # Number of "Trees" in the forest
    max_depth=10,          # Don't let it over-memorize
    random_state=42
)
model.fit(X_train, y_train)

# --- 3. EVALUATE PERFORMANCE ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*30)
print(f"🏆 MODEL ACCURACY: {accuracy:.1%}")
print("="*30)
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

# --- 4. EXPLAINABILITY (Feature Importance) ---
importances = model.feature_importances_
feature_names = X.columns

# Create a DataFrame for plotting
fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
fi_df = fi_df.sort_values(by='Importance', ascending=False)

print("\n🔍 What drives Conflict? (Feature Importance):")
print(fi_df)

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis')
plt.title('What Environmental Factors Cause Conflict?', fontsize=16)
plt.xlabel('Importance Score (0-1)')
plt.tight_layout()
plt.savefig(PLOT_FILE)
print(f"📊 Feature Importance Graph saved to '{PLOT_FILE}'")

# --- 5. SAVE THE MODEL ---
joblib.dump(model, MODEL_FILE)
print(f"💾 Trained Model saved to '{MODEL_FILE}'")
print("✅ Ready to be used in app.py!")