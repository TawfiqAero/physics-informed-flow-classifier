# ==============================
# 1. IMPORT LIBRARIES
# ==============================
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ==============================
# 2. LOAD DATASET
# ==============================
df = pd.read_csv("airfoil_dataset_10k.csv")

# Features and target
X = df[["Reynolds", "AoA", "Cl", "Cd"]]
y = df["Flow_Regime"]

print("Dataset Loaded:", df.shape)


# ==============================
# 3. TRAIN-TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ==============================
# 4. MODEL + HYPERPARAMETER TUNING
# ==============================
param_grid = {
    "n_estimators": [100, 150],
    "max_depth": [6, 8, 10],
    "min_samples_split": [2, 5]
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

model = grid_search.best_estimator_

print("\nBest Parameters:", grid_search.best_params_)


# ==============================
# 5. EVALUATE MODEL
# ==============================
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))


# ==============================
# 6. FEATURE IMPORTANCE (VERY IMPORTANT)
# ==============================
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance = feature_importance.sort_values(ascending=False)

print("\nFeature Importance:\n")
print(feature_importance)


# ==============================
# 7. SAVE MODEL
# ==============================
with open("flow_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅ Model saved as flow_model.pkl")
