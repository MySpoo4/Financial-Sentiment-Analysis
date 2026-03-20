import classifiers.bayesian_classifier as bayes
import classifiers.logistic_regression as logistic
import classifiers.random_forest as forest
from dataset import x, y


def train_all():
    print("Training Bayesian Classifier")
    bayes.train_model(x, y)

    print("Training Logistic Regression Classifier")
    logistic.train_model(x, y)

    print("Training Random Forest Classifier")
    forest.train_model(x, y)


if __name__ == "__main__":
    train_all()
