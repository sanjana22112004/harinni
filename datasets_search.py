import streamlit as st
import openml
import pandas as pd
from datasets import load_dataset

@st.cache_data
def load_openml_dataset(dataset_id):
    try:
        dataset = openml.datasets.get_dataset(int(dataset_id))
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        df = pd.concat([X, y], axis=1)
        # Limit the dataset size to 5000 rows to prevent resource overload
        if len(df) > 5000:
            df = df.sample(n=5000, random_state=42).reset_index(drop=True)
        return df
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})

@st.cache_data
def load_huggingface_dataset(name):
    try:
        dataset = load_dataset(name)
        df = dataset['train'].to_pandas()
        return df.head(2000)
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})
