import joblib
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt

# 1. Load the winning model you already saved!
# (Make sure to run this on your Logistic Regression or Random Forest .pkl)
model = joblib.load("./models/logistic_regression.pkl")

# 2. Pick a tricky or interesting headline to analyze
# You can pull a real one from your 'x' dataset
text_to_test = "Operating profit fell 15% but the company expects strong revenue growth next quarter."

# 3. Setup the LIME Explainer
# Make sure the class names match the order of your classes (usually alphabetical: Negative, Neutral, Positive)
explainer = LimeTextExplainer(class_names=["Negative", "Neutral", "Positive"])

# 4. Generate the Explanation
# LIME tweaks the sentence hundreds of times and watches how predict_proba changes
exp = explainer.explain_instance(
    text_to_test,
    model.predict_proba,
    num_features=6,  # How many words to highlight
    top_labels=1,  # Only explain the winning class
)

# 5. Output the results
winning_class_index = list(exp.local_exp.keys())[0]
print(f"The model predicted class index: {winning_class_index}")
print("Word Importance Breakdown:")

# CHANGED: Added 'label=winning_class_index' inside the parentheses
for word, weight in exp.as_list(label=winning_class_index):
    print(f"{word}: {weight:.4f}")

# 6. Create the Visual Chart
# (This one already had the label parameter, but just double-checking!)
fig = exp.as_pyplot_figure(label=winning_class_index)
import matplotlib.pyplot as plt  # Make sure plt is imported to show the graph

plt.tight_layout()
plt.show()
