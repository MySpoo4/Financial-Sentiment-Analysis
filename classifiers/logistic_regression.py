from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
from dataset import x, y

# 1. Pipeline
pipeline = Pipeline(
    [
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ]
)

# 2. Define the Hyperparameters
param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],  # Try single words vs. phrases
    "tfidf__max_features": [2000, 5000],  # Try limiting vocabulary size
    "clf__C": [0.1, 1.0, 10.0],  # Try different regularization (strictness)
}

# 3. Setup the 10-Fold Strategy
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 4. Run the Grid Search Competition
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1)
grid_search.fit(x, y)

# 6. Save Winner
model = grid_search.best_estimator_
joblib.dump(model, "../models/logistic_regression.pkl")

# 7. Final Evalaution
y_pred = cross_val_predict(model, x, y, cv=cv)

print("\n--- FINAL CLASSIFICATION REPORT (10-Fold CV) ---")
print(classification_report(y, y_pred))
