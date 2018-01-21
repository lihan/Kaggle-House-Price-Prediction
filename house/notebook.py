import pandas as pd
from sklearn.model_selection import train_test_split

train_dataframe = pd.read_csv('../input/train.csv")')
X_train, X_test, y_train, y_test = train_test_split(train_dataframe)