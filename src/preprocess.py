import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    df = df.dropna()
    
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Identify the salary column dynamically
    target_col = [col for col in df.columns if 'salary' in col.lower()]
    if not target_col:
        raise ValueError("No column containing 'salary' found in the dataset.")

    target_col = target_col[0]  # take the first match

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    return X, y, label_encoders
