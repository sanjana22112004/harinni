import streamlit as st
import pandas as pd
from datasets_search import load_openml_dataset, load_huggingface_dataset
from preprocessing import preprocess_data
from models import train_and_evaluate
from utils import plot_correlation_matrix
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ML Playground", layout="wide")

st.title("🚀 Machine Learning Playground")
st.markdown("---")

# Dataset selection
st.sidebar.header("📂 Dataset Options")
dataset_source = st.sidebar.radio("Choose dataset source:", ["Upload CSV", "OpenML", "Hugging Face"])

df = None

if dataset_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

elif dataset_source == "OpenML":
    openml_id = st.sidebar.text_input("Enter OpenML dataset ID", "61")
    st.sidebar.caption("👉 Example: 61 = Iris dataset")
    if st.sidebar.button("Load from OpenML"):
        df = load_openml_dataset(openml_id)

elif dataset_source == "Hugging Face":
    hf_name = st.sidebar.text_input("Enter Hugging Face dataset name", "imdb")
    st.sidebar.caption("👉 Example: imdb = sentiment analysis dataset")
    if st.sidebar.button("Load from Hugging Face"):
        df = load_huggingface_dataset(hf_name)

if df is not None:
    st.write("### 📊 Dataset Preview", df.head())
    st.write("Shape:", df.shape)

    target_col = st.selectbox("🎯 Select target column", df.columns)
    if target_col:
        st.markdown("---")
        st.header("⚙️ Data Processing & Modeling")
        
        # Determine if it's a classification or regression problem
        is_classification = len(df[target_col].unique()) < 20 and df[target_col].dtype in ['object', 'int64']

        X_train, X_test, y_train, y_test = preprocess_data(df, target_col)

        st.subheader("Model Performance")

        with st.spinner('Training models...'):
            results, trained_models = train_and_evaluate(X_train, X_test, y_train, y_test)
        
        # Display results
        for name, metrics in results.items():
            st.write(f"#### {name}")
            if is_classification:
                st.metric("Accuracy", f"{metrics['accuracy']:.2f}")
                
                # Plot confusion matrix
                fig, ax = plt.subplots(figsize=(6, 4))
                cm = confusion_matrix(y_test, metrics['predictions'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                st.pyplot(fig)
                st.text(classification_report(y_test, metrics['predictions']))
            else:
                st.metric("R² Score", f"{metrics['R2']:.2f}")
                st.metric("Mean Squared Error (MSE)", f"{metrics['MSE']:.2f}")
                st.metric("Mean Absolute Error (MAE)", f"{metrics['MAE']:.2f}")

        # Correlation matrix visualization
        st.markdown("---")
        st.header("📈 Data Visualization")
        st.subheader("Correlation Matrix")
        fig = plot_correlation_matrix(df)
        st.pyplot(fig)

        st.markdown("---")
        st.header("🔮 Model Prediction")
        
        # Prediction interface
        model_name = st.selectbox("Select a trained model for prediction:", list(trained_models.keys()))
        selected_model = trained_models[model_name]
        
        input_data = {}
        st.subheader("Enter values for prediction:")
        for col in X_train.columns:
            if X_train[col].dtype == 'object':
                unique_vals = X_train[col].unique()
                input_data[col] = st.selectbox(f"Select value for {col}", unique_vals)
            else:
                min_val = float(X_train[col].min())
                max_val = float(X_train[col].max())
                input_data[col] = st.number_input(f"Enter value for {col}", min_value=min_val, max_value=max_val, value=float(X_train[col].median()))

        if st.button("Make Prediction"):
            input_df = pd.DataFrame([input_data])
            if is_classification:
                pred = selected_model.predict(input_df)[0]
                st.success(f"Prediction: {pred}")
            else:
                pred = selected_model.predict(input_df)[0]
                st.success(f"Prediction: {pred:.3f}")
