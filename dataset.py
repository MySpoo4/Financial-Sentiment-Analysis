import kagglehub
from kagglehub import KaggleDatasetAdapter

# path to csv file
file_path = "data.csv"

# Load the latest version
# https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis
df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "sbhatti/financial-sentiment-analysis",
    file_path,
)

x = df["Sentence"]
y = df["Sentiment"]
