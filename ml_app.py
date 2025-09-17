import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.metrics import roc_curve, auc, confusion_matrix, r2_score
import openml
from datasets import load_dataset
from scipy import stats

# Set page config
st.set_page_config(page_title="ML Playground", layout="wide")

# Title
st.title("🚀 Machine Learning Playground")
st.markdown("**Advanced ML with Comprehensive EDA & Model Evaluation**")

# Dataset loading functions
def load_openml_dataset(dataset_id):
    try:
        dataset = openml.datasets.get_dataset(int(dataset_id))
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        
        # Handle different data types and ensure proper DataFrame creation
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X)
        
        if isinstance(y, pd.Series):
            df[dataset.default_target_attribute] = y
        else:
            df[dataset.default_target_attribute] = pd.Series(y)
        
        # Clean column names (remove special characters)
        df.columns = [str(col).replace(' ', '_').replace('-', '_') for col in df.columns]
        
        # Remove any completely empty columns
        df = df.dropna(axis=1, how='all')
        
        return df
        
    except Exception as e:
        st.error(f"Failed to load OpenML dataset {dataset_id}: {str(e)}")
        return None

def load_huggingface_dataset(name):
    try:
        dataset = load_dataset(name)
        df = dataset['train'].to_pandas()
        return df.head(2000)
    except Exception as e:
        st.error(f"Failed to load Hugging Face dataset {name}: {str(e)}")
        return None

# Preprocessing function
def preprocess_data(df, target_col):
    """Enhanced preprocessing with better error handling"""
    try:
        # Create a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Remove rows where target is NaN
        df_processed = df_processed.dropna(subset=[target_col])
        
        # Separate features and target
        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]
        
        # Handle missing values in features
        # For numeric columns, use median imputation
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_imputer = SimpleImputer(strategy='median')
            X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])
        
        # For categorical columns, use most frequent imputation
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            X[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])
        
        # Encode categorical features
        for col in categorical_cols:
            if col in X.columns:  # Check if column still exists after imputation
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Encode target variable if it's categorical
        if y.dtype == 'object' or y.dtype.name == 'category':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))
        
        # Ensure all data is numeric
        X = X.astype(float)
        y = y.astype(float)
        
        # Remove any remaining NaN values
        X = X.fillna(X.median())
        
        # Check if we have valid data
        if X.empty or len(X) == 0:
            raise ValueError("No valid data after preprocessing")
        
        # Ensure we have at least 2 samples for train/test split
        if len(X) < 2:
            raise ValueError("Not enough data for train/test split")
        
        return train_test_split(X, y, test_size=0.2, random_state=42)
        
    except Exception as e:
        # Fallback to simple preprocessing
        st.warning(f"Advanced preprocessing failed, using fallback: {e}")
        
        # Simple fallback
        df_processed = df.dropna()
        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]
        
        # Simple encoding
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y.astype(str))
        
        # Ensure numeric types
        X = X.astype(float)
        y = y.astype(float)
        
        return train_test_split(X, y, test_size=0.2, random_state=42)

# Model training function
def train_and_evaluate(X_train, X_test, y_train, y_test):
    results = {}
    trained_models = {}
    is_classification = len(set(y_train)) < 20

    if is_classification:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=500),
            "Random Forest": RandomForestClassifier(random_state=42)
        }
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            pred_proba = model.predict_proba(X_test)
            acc = accuracy_score(y_test, preds)
            
            results[name] = {
                "accuracy": acc,
                "predictions": preds,
                "probabilities": pred_proba,
                "model": model
            }
            trained_models[name] = model
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(random_state=42)
        }
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            
            results[name] = {
                "MSE": mse, 
                "RMSE": np.sqrt(mse),
                "predictions": preds,
                "model": model
            }
            trained_models[name] = model
    
    return results, trained_models

# Visualization functions
def plot_correlation_matrix(df):
    """📊 Correlation Heatmap for numeric features"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return None
    
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, 
                square=True, cbar_kws={"shrink": .8})
    plt.title("📊 Correlation Heatmap", fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_feature_distributions(df):
    """📈 Feature Distribution plots (histograms for numeric, bar plots for categorical)"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Calculate subplot dimensions
    total_cols = len(numeric_cols) + len(categorical_cols)
    if total_cols == 0:
        return None
    
    n_cols = min(3, total_cols)
    n_rows = (total_cols + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    plot_idx = 0
    
    # Plot numeric features as histograms
    for col in numeric_cols:
        if plot_idx < len(axes):
            axes[plot_idx].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[plot_idx].set_title(f"📊 {col} (Numeric)", fontweight='bold')
            axes[plot_idx].set_xlabel(col)
            axes[plot_idx].set_ylabel("Frequency")
            plot_idx += 1
    
    # Plot categorical features as bar plots
    for col in categorical_cols:
        if plot_idx < len(axes):
            value_counts = df[col].value_counts().head(10)  # Top 10 categories
            axes[plot_idx].bar(range(len(value_counts)), value_counts.values)
            axes[plot_idx].set_title(f"📊 {col} (Categorical)", fontweight='bold')
            axes[plot_idx].set_xlabel(col)
            axes[plot_idx].set_ylabel("Count")
            axes[plot_idx].set_xticks(range(len(value_counts)))
            axes[plot_idx].set_xticklabels(value_counts.index, rotation=45, ha='right')
            plot_idx += 1
    
    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle("📈 Feature Distributions", fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_boxplots_numeric_vs_target(df, target_col):
    """📉 Boxplots for numeric features vs target (if target is numeric)"""
    if target_col not in df.columns:
        return None
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != target_col]
    
    if len(numeric_cols) == 0 or df[target_col].dtype not in [np.number]:
        return None
    
    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            sns.boxplot(data=df, x=target_col, y=col, ax=axes[i])
            axes[i].set_title(f"📉 {col} vs {target_col}", fontweight='bold')
            axes[i].tick_params(axis='x', rotation=45)
    
    # Hide unused subplots
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f"📉 Numeric Features vs Target ({target_col})", fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_roc_curve(y_true, y_pred_proba, model_name="Model"):
    """✅ ROC Curve for binary classification"""
    if len(np.unique(y_true)) != 2:
        return None
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'✅ ROC Curve - {model_name}', fontsize=16, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
    """✅ Confusion Matrix for classification"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'✅ Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_feature_importance(model, feature_names, model_name="Model", top_n=10):
    """✅ Feature Importance for tree-based models"""
    if not hasattr(model, 'feature_importances_'):
        return None
    
    importances = model.feature_importances_
    n_features = len(importances)
    
    # Adjust top_n if there are fewer features
    actual_top_n = min(top_n, n_features)
    indices = np.argsort(importances)[::-1][:actual_top_n]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(actual_top_n), importances[indices])
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    ax.set_title(f'✅ Feature Importance - {model_name}', fontsize=16, fontweight='bold')
    ax.set_xticks(range(actual_top_n))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_residuals(y_true, y_pred, model_name="Model"):
    """✅ Residual Plot for regression"""
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Residuals vs Predicted
    ax1.scatter(y_pred, residuals, alpha=0.6)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title(f'✅ Residuals vs Predicted - {model_name}', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot for residuals
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title(f'Q-Q Plot of Residuals - {model_name}', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_prediction_probabilities(y_pred_proba, model_name="Model"):
    """📊 Probability Distribution for Classification Predictions"""
    if y_pred_proba.ndim == 1:
        # Binary classification
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(y_pred_proba, bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Frequency')
        ax.set_title(f'📊 Probability Distribution - {model_name}', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
    else:
        # Multi-class classification
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(y_pred_proba.shape[1]):
            ax.hist(y_pred_proba[:, i], bins=30, alpha=0.6, label=f'Class {i}')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Frequency')
        ax.set_title(f'📊 Probability Distribution - {model_name}', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_actual_vs_predicted(y_true, y_pred, model_name="Model"):
    """📊 Scatter plot of actual vs predicted for regression"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.6)
    
    # Perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'📊 Actual vs Predicted - {model_name}', fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add R² score
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    return fig

# Main App Interface
# Dataset selection
st.sidebar.header("📂 Dataset Options")
dataset_source = st.sidebar.radio("Choose dataset source:", ["Upload CSV", "OpenML", "Hugging Face"])

# Add info about dataset sources
with st.sidebar.expander("ℹ️ Dataset Info"):
    st.write("""
    **📁 Upload CSV**: Upload your own dataset
    
    **🔬 OpenML**: 1000+ datasets for ML research
    - Classification: Iris, Wine, Breast Cancer, etc.
    - Regression: Boston Housing, Auto MPG, etc.
    
    **🤗 Hugging Face**: NLP and text datasets
    - Sentiment: IMDB, Amazon, Yelp reviews
    - Classification: News, DBpedia
    - GLUE benchmark tasks
    """)

df = None

if dataset_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

elif dataset_source == "OpenML":
    st.sidebar.subheader("📋 Popular Datasets")
    
    # Popular OpenML datasets
    popular_datasets = {
        "Iris (Classification)": "61",
        "Wine (Classification)": "187", 
        "Breast Cancer (Classification)": "13",
        "Diabetes (Classification)": "37",
        "Heart Disease (Classification)": "45",
        "Boston Housing (Regression)": "529",
        "Auto MPG (Regression)": "9",
        "Abalone (Regression)": "183",
        "CPU Performance (Regression)": "562",
        "Servo (Regression)": "871",
        "Glass Identification (Classification)": "40",
        "Sonar (Classification)": "151",
        "Vehicle (Classification)": "54",
        "Segment (Classification)": "36",
        "Waveform (Classification)": "60"
    }
    
    # Dataset selection method
    selection_method = st.sidebar.radio("Choose selection method:", ["Popular Datasets", "Custom ID"])
    
    if selection_method == "Popular Datasets":
        selected_dataset = st.sidebar.selectbox(
            "Select a dataset:",
            options=list(popular_datasets.keys()),
            index=0
        )
        openml_id = popular_datasets[selected_dataset]
        st.sidebar.caption(f"Dataset ID: {openml_id}")
    else:
        openml_id = st.sidebar.text_input("Enter OpenML dataset ID", "61")
        st.sidebar.caption("👉 Example: 61 = Iris dataset")
    
    if st.sidebar.button("Load from OpenML"):
        with st.spinner(f"🔄 Loading dataset {openml_id}..."):
            df = load_openml_dataset(openml_id)
            if df is None:
                st.error("❌ Failed to load dataset from OpenML")
            else:
                st.success(f"✅ Successfully loaded dataset {openml_id}")

elif dataset_source == "Hugging Face":
    st.sidebar.subheader("📋 Popular NLP Datasets")
    
    # Popular Hugging Face datasets
    popular_hf_datasets = {
        "IMDB Reviews (Sentiment)": "imdb",
        "Amazon Reviews (Sentiment)": "amazon_polarity",
        "Yelp Reviews (Sentiment)": "yelp_review_full",
        "AG News (Classification)": "ag_news",
        "DBpedia (Classification)": "dbpedia_14",
        "20 Newsgroups (Classification)": "newsgroup",
        "SQuAD (QA)": "squad",
        "CoLA (Grammar)": "glue",
        "SST-2 (Sentiment)": "glue",
        "MRPC (Paraphrase)": "glue",
        "QQP (Paraphrase)": "glue",
        "MNLI (NLI)": "glue",
        "QNLI (NLI)": "glue",
        "RTE (NLI)": "glue",
        "WNLI (NLI)": "glue"
    }
    
    # Dataset selection method
    hf_selection_method = st.sidebar.radio("Choose selection method:", ["Popular Datasets", "Custom Name"])
    
    if hf_selection_method == "Popular Datasets":
        selected_hf_dataset = st.sidebar.selectbox(
            "Select a dataset:",
            options=list(popular_hf_datasets.keys()),
            index=0
        )
        hf_name = popular_hf_datasets[selected_hf_dataset]
        st.sidebar.caption(f"Dataset: {hf_name}")
    else:
        hf_name = st.sidebar.text_input("Enter Hugging Face dataset name", "imdb")
        st.sidebar.caption("👉 Example: imdb = sentiment analysis dataset")
    
    if st.sidebar.button("Load from Hugging Face"):
        with st.spinner(f"🔄 Loading dataset {hf_name}..."):
            df = load_huggingface_dataset(hf_name)
            if df is None:
                st.error("❌ Failed to load dataset from Hugging Face")
            else:
                st.success(f"✅ Successfully loaded dataset {hf_name}")

if df is not None and not df.empty:
    st.write("### 📊 Dataset Preview")
    st.dataframe(df.head(10))
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Dataset info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Features", len(df.columns))
    with col2:
        st.metric("Numeric Features", len(df.select_dtypes(include=[np.number]).columns))
    with col3:
        st.metric("Categorical Features", len(df.select_dtypes(include=['object', 'category']).columns))

    target_col = st.selectbox("🎯 Select target column", df.columns)
    
    if target_col:
        # Determine problem type
        unique_targets = df[target_col].nunique()
        is_classification = unique_targets < 20
        
        st.info(f"**Problem Type:** {'Classification' if is_classification else 'Regression'} ({unique_targets} unique values)")
        
        # Preprocessing
        try:
            with st.spinner("🔄 Preprocessing data..."):
                X_train, X_test, y_train, y_test = preprocess_data(df, target_col)
            
            # Check if preprocessing was successful
            if X_train is None or len(X_train) == 0:
                st.error("❌ Preprocessing failed: No valid data after preprocessing")
                st.stop()
            
            st.success(f"✅ Data preprocessed successfully: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
            
        except Exception as e:
            st.error(f"❌ Preprocessing failed: {str(e)}")
            st.info("💡 Try selecting a different target column or check your data for issues")
            st.stop()
        
        # Model Training
        try:
            with st.spinner("🤖 Training models..."):
                results, trained_models = train_and_evaluate(X_train, X_test, y_train, y_test)
            
            if not results or not trained_models:
                st.error("❌ Model training failed")
                st.stop()
                
        except Exception as e:
            st.error(f"❌ Model training failed: {str(e)}")
            st.info("💡 This might be due to data type issues or insufficient data")
            st.stop()
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🤖 Model Results", "📈 Predictions", "ℹ️ Dataset Info"])
        
        with tab1:
            st.header("📊 Exploratory Data Analysis (EDA)")
            
            # Correlation Matrix
            st.subheader("📊 Correlation Heatmap")
            corr_plot = plot_correlation_matrix(df)
            if corr_plot:
                st.pyplot(corr_plot)
            else:
                st.warning("Not enough numeric features for correlation analysis")
            
            # Feature Distributions
            st.subheader("📈 Feature Distributions")
            dist_plot = plot_feature_distributions(df)
            if dist_plot:
                st.pyplot(dist_plot)
            else:
                st.warning("No features available for distribution analysis")
            
            # Boxplots for numeric vs target
            if is_classification:
                st.subheader("📉 Numeric Features vs Target")
                box_plot = plot_boxplots_numeric_vs_target(df, target_col)
                if box_plot:
                    st.pyplot(box_plot)
                else:
                    st.warning("Target column is not numeric or no numeric features available")
        
        with tab2:
            st.header("🤖 Model Results & Evaluation")
            
            # Model Performance Metrics
            st.subheader("📊 Performance Metrics")
            for model_name, metrics in results.items():
                with st.expander(f"**{model_name}**"):
                    if is_classification:
                        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("MSE", f"{metrics['MSE']:.3f}")
                        with col2:
                            st.metric("RMSE", f"{metrics['RMSE']:.3f}")
            
            # Model Evaluation Plots
            st.subheader("✅ Model Evaluation Visualizations")
            
            for model_name, metrics in results.items():
                st.write(f"#### {model_name}")
                
                if is_classification:
                    # ROC Curve
                    if 'probabilities' in metrics:
                        roc_plot = plot_roc_curve(y_test, metrics['probabilities'][:, 1], model_name)
                        if roc_plot:
                            st.pyplot(roc_plot)
                    
                    # Confusion Matrix
                    conf_plot = plot_confusion_matrix(y_test, metrics['predictions'], model_name)
                    if conf_plot:
                        st.pyplot(conf_plot)
                    
                    # Feature Importance (for tree-based models)
                    if hasattr(metrics['model'], 'feature_importances_'):
                        feat_imp_plot = plot_feature_importance(
                            metrics['model'], X_train.columns, model_name
                        )
                        if feat_imp_plot:
                            st.pyplot(feat_imp_plot)
                
                else:
                    # Residual Plot for regression
                    resid_plot = plot_residuals(y_test, metrics['predictions'], model_name)
                    if resid_plot:
                        st.pyplot(resid_plot)
                    
                    # Feature Importance (for tree-based models)
                    if hasattr(metrics['model'], 'feature_importances_'):
                        feat_imp_plot = plot_feature_importance(
                            metrics['model'], X_train.columns, model_name
                        )
                        if feat_imp_plot:
                            st.pyplot(feat_imp_plot)
        
        with tab3:
            st.header("📈 Predictions")
            
            # Model Selection for Predictions
            model_names = list(trained_models.keys())
            selected_model_name = st.selectbox("Select model for predictions:", model_names)
            selected_model = trained_models[selected_model_name]
            
            # Prediction Type Selection
            pred_type = st.radio("Choose prediction type:", ["Manual Input", "Batch Upload"])
            
            if pred_type == "Manual Input":
                st.subheader("🔮 Manual Predictions")
                st.write("Enter feature values for prediction:")
                
                # Create input form
                input_data = {}
                feature_cols = X_train.columns
                
                cols = st.columns(min(3, len(feature_cols)))
                for i, col in enumerate(feature_cols):
                    with cols[i % len(cols)]:
                        if X_train[col].dtype in ['int64', 'float64']:
                            # Numeric input
                            min_val = float(X_train[col].min())
                            max_val = float(X_train[col].max())
                            input_data[col] = st.number_input(
                                f"{col}", 
                                min_value=min_val, 
                                max_value=max_val, 
                                value=float(X_train[col].median())
                            )
                        else:
                            # Categorical input
                            unique_vals = X_train[col].unique()
                            input_data[col] = st.selectbox(f"{col}", unique_vals)
                
                if st.button("🔮 Make Prediction"):
                    # Convert to DataFrame
                    input_df = pd.DataFrame([input_data])
                    
                    # Make prediction
                    if is_classification:
                        pred = selected_model.predict(input_df)[0]
                        pred_proba = selected_model.predict_proba(input_df)[0]
                        
                        st.success(f"**Prediction:** {pred}")
                        st.write("**Class Probabilities:**")
                        classes = selected_model.classes_
                        for i, prob in enumerate(pred_proba):
                            st.write(f"  {classes[i]}: {prob:.3f}")
                    else:
                        pred = selected_model.predict(input_df)[0]
                        st.success(f"**Prediction:** {pred:.3f}")
            
            else:  # Batch Upload
                st.subheader("📁 Batch Predictions")
                uploaded_pred_file = st.file_uploader("Upload CSV for batch predictions", type=["csv"])
                
                if uploaded_pred_file is not None:
                    pred_df = pd.read_csv(uploaded_pred_file)
                    st.write("**Uploaded Data Preview:**")
                    st.dataframe(pred_df.head())
                    
                    if st.button("🔮 Make Batch Predictions"):
                        # Ensure same features as training data
                        missing_cols = set(X_train.columns) - set(pred_df.columns)
                        if missing_cols:
                            st.error(f"Missing columns: {missing_cols}")
                        else:
                            # Make predictions
                            pred_df_subset = pred_df[X_train.columns]
                            predictions = selected_model.predict(pred_df_subset)
                            
                            # Add predictions to dataframe
                            result_df = pred_df.copy()
                            result_df['Prediction'] = predictions
                            
                            if is_classification:
                                probabilities = selected_model.predict_proba(pred_df_subset)
                                for i, class_name in enumerate(selected_model.classes_):
                                    result_df[f'Probability_{class_name}'] = probabilities[:, i]
                            
                            st.write("**Prediction Results:**")
                            st.dataframe(result_df)
                            
                            # Download results
                            csv = result_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Predictions",
                                data=csv,
                                file_name="predictions.csv",
                                mime="text/csv"
                            )
            
            # Prediction Visualizations
            st.subheader("📊 Prediction Visualizations")
            
            if is_classification and 'probabilities' in results[selected_model_name]:
                # Probability Distribution
                prob_plot = plot_prediction_probabilities(
                    results[selected_model_name]['probabilities'], selected_model_name
                )
                if prob_plot:
                    st.pyplot(prob_plot)
                
                # ROC Curve for predictions
                roc_plot = plot_roc_curve(
                    y_test, results[selected_model_name]['probabilities'][:, 1], selected_model_name
                )
                if roc_plot:
                    st.pyplot(roc_plot)
            
            else:
                # Actual vs Predicted for regression
                actual_pred_plot = plot_actual_vs_predicted(
                    y_test, results[selected_model_name]['predictions'], selected_model_name
                )
                if actual_pred_plot:
                    st.pyplot(actual_pred_plot)
        
        with tab4:
            st.header("ℹ️ Dataset Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Data Types")
                dtype_df = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes,
                    'Non-Null Count': df.count(),
                    'Null Count': df.isnull().sum()
                })
                st.dataframe(dtype_df)
            
            with col2:
                st.subheader("📈 Statistical Summary")
                st.dataframe(df.describe())
            
            st.subheader("🔍 Missing Values")
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0]
            if len(missing_data) > 0:
                st.bar_chart(missing_data)
            else:
                st.success("✅ No missing values found!")
            
            st.subheader("🎯 Target Variable Analysis")
            st.write(f"**Target Column:** {target_col}")
            st.write(f"**Unique Values:** {df[target_col].nunique()}")
            st.write(f"**Data Type:** {df[target_col].dtype}")
            
            if is_classification:
                st.write("**Class Distribution:**")
                class_counts = df[target_col].value_counts()
                st.bar_chart(class_counts)
            else:
                st.write("**Target Statistics:**")
                st.write(df[target_col].describe())



