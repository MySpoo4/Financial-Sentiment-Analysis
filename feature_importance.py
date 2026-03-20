import joblib
import pandas as pd

# ==========================================
# 1. LOGISTIC REGRESSION (Coefficients)
# ==========================================
print("==========================================")
print("       LOGISTIC REGRESSION INSIGHTS       ")
print("==========================================")
lr_model = joblib.load("./models/logistic_regression.pkl")
lr_vect = lr_model.named_steps["tfidf"]
lr_clf = lr_model.named_steps["clf"]
lr_features = lr_vect.get_feature_names_out()

for i, class_label in enumerate(lr_clf.classes_):
    print(f"\n--- Top 10 Keywords driving the '{class_label}' class ---")
    # Logistic Regression uses .coef_
    coef_df = pd.DataFrame(
        {"Keyword": lr_features, "Importance Score": lr_clf.coef_[i]}
    )
    top_words = coef_df.sort_values(by="Importance Score", ascending=False).head(10)
    print(top_words.to_string(index=False))

# ==========================================
# 2. NAIVE BAYES (Log Probabilities)
# ==========================================
print("\n\n==========================================")
print("          NAIVE BAYES INSIGHTS            ")
print("==========================================")
nb_model = joblib.load("./models/bayesian_classifier.pkl")
# Remember: NB used CountVectorizer named 'vect' and MultinomialNB named 'nb'
nb_vect = nb_model.named_steps["vect"]
nb_clf = nb_model.named_steps["nb"]
nb_features = nb_vect.get_feature_names_out()

for i, class_label in enumerate(nb_clf.classes_):
    print(f"\n--- Top 10 Keywords driving the '{class_label}' class ---")
    # Naive Bayes uses .feature_log_prob_
    # (These are negative numbers; values closer to 0 mean higher probability)
    prob_df = pd.DataFrame(
        {"Keyword": nb_features, "Log Probability": nb_clf.feature_log_prob_[i]}
    )
    top_words = prob_df.sort_values(by="Log Probability", ascending=False).head(10)
    print(top_words.to_string(index=False))

# ==========================================
# 3. RANDOM FOREST (Gini Importance)
# ==========================================
print("\n\n==========================================")
print("         RANDOM FOREST INSIGHTS           ")
print("==========================================")
rf_model = joblib.load("./models/random_forest.pkl")
rf_vect = rf_model.named_steps["tfidf"]
rf_clf = rf_model.named_steps["rf"]
rf_features = rf_vect.get_feature_names_out()

print(f"\n--- Top 15 Keywords driving the ENTIRE Random Forest ---")
print("(Note: Trees calculate global importance, not per-class importance)")
# Random Forest uses .feature_importances_ (Just one list, not separated by class)
rf_df = pd.DataFrame(
    {"Keyword": rf_features, "Gini Importance": rf_clf.feature_importances_}
)
top_rf_words = rf_df.sort_values(by="Gini Importance", ascending=False).head(15)
print(top_rf_words.to_string(index=False))

# import joblib
# import pandas as pd
#
# # 1. Load the winning Logistic Regression model
# model = joblib.load("./models/logistic_regression.pkl")
#
# # 2. Extract the two pieces of the puzzle
# # 'tfidf' holds the words, 'clf' holds the math
# vectorizer = model.named_steps["tfidf"]
# classifier = model.named_steps["clf"]
#
# # 3. Get the actual list of words (the vocabulary)
# feature_names = vectorizer.get_feature_names_out()
#
# # 4. Loop through each class (Negative, Neutral, Positive)
# # classifier.classes_ tells us the exact names of your classes
# for i, class_label in enumerate(classifier.classes_):
#     print(f"\n--- Top 10 Keywords driving the '{class_label}' class ---")
#
#     # Get the coefficients for this specific class
#     coefficients = classifier.coef_[i]
#
#     # Zip the words and their scores together into a DataFrame
#     coef_df = pd.DataFrame({"Keyword": feature_names, "Importance Score": coefficients})
#
#     # Sort them by the highest score to find the strongest drivers
#     top_words = coef_df.sort_values(by="Importance Score", ascending=False).head(10)
#     print(top_words.to_string(index=False))
