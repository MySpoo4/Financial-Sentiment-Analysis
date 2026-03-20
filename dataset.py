# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = "data.csv"

# Load the latest version
# https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis
df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "sbhatti/financial-sentiment-analysis",
    file_path,
    # Provide any additional arguments like
    # sql_query or pandas_kwargs. See the
    # documenation for more information:
    # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

x = df["Sentence"]
y = df["Sentiment"]
