from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
from dataset import x, y

# 1. Pipeline
pipeline = Pipeline(
    [
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("rf", RandomForestClassifier(class_weight="balanced", random_state=42)),
    ]
)

# 2. Define the Hyperparameters
param_grid = {
    "tfidf__max_features": [500, 1000, 2000],
    "rf__n_estimators": [300],
    "rf__max_depth": [10, 20, None],
    "rf__min_samples_leaf": [2, 5, 10],
    "rf__max_samples": [0.8],
}

# 3. Setup the 10-Fold Strategy
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 4. Run GridSearch to find the best settings
grid_search = GridSearchCV(
    pipeline, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=1
)
grid_search.fit(x, y)

# 6. Save Winner
model = grid_search.best_estimator_
joblib.dump(model, "../models/random_forest.pkl")

# 7. Final Evalaution
y_pred = cross_val_predict(model, x, y, cv=cv)

print("\n--- FINAL CLASSIFICATION REPORT (10-Fold CV) ---")
print(classification_report(y, y_pred))
