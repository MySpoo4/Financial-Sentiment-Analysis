from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import os


def train_model(x, y):
    # 1. Pipeline
    pipeline = Pipeline(
        [("vect", CountVectorizer(stop_words="english")), ("nb", MultinomialNB())]
    )

    # 2. Define the Hyperparameters
    param_grid = {
        "vect__ngram_range": [(1, 1), (1, 2)],
        "nb__alpha": [0.1, 0.5, 1.0],
    }

    # 3. Setup the 10-Fold Strategy
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # 4. Run GridSearch to find the best settings
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1
    )
    grid_search.fit(x, y)

    # 6. Save Winner
    model = grid_search.best_estimator_

    if not os.path.exists("./models"):
        os.makedirs("./models")

    joblib.dump(model, "./models/bayesian_classifier.pkl")

    # 7. Final Evaluation
    y_pred = cross_val_predict(model, x, y, cv=cv)

    print("\n--- FINAL CLASSIFICATION REPORT (10-Fold CV) ---")
    print(classification_report(y, y_pred))
