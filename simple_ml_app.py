import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, roc_curve, auc, confusion_matrix, r2_score

# Set page config
st.set_page_config(page_title="ML Playground", layout="wide")

# Title
st.title("🚀 Machine Learning Playground")
st.markdown("**Simple ML App with Visualizations**")

# Sample data generation function
def create_sample_data():
    """Create sample datasets for demonstration"""
    np.random.seed(42)
    
    # Sample classification data (Iris-like)
    n_samples = 150
    data = {
        'sepal_length': np.random.normal(5.8, 0.8, n_samples),
        'sepal_width': np.random.normal(3.0, 0.4, n_samples),
        'petal_length': np.random.normal(3.8, 1.8, n_samples),
        'petal_width': np.random.normal(1.2, 0.8, n_samples),
        'species': np.random.choice(['setosa', 'versicolor', 'virginica'], n_samples)
    }
    
    return pd.DataFrame(data)

# Visualization functions
def plot_correlation_matrix(df):
    """Correlation Heatmap"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return None
    
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    return fig

def plot_feature_distributions(df):
    """Feature Distribution plots"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    total_cols = len(numeric_cols) + len(categorical_cols)
    if total_cols == 0:
        return None
    
    n_cols = min(3, total_cols)
    n_rows = (total_cols + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    plot_idx = 0
    
    # Plot numeric features
    for col in numeric_cols:
        if plot_idx < len(axes):
            axes[plot_idx].hist(df[col].dropna(), bins=20, alpha=0.7)
            axes[plot_idx].set_title(f"{col} (Numeric)")
            axes[plot_idx].set_xlabel(col)
            axes[plot_idx].set_ylabel("Frequency")
            plot_idx += 1
    
    # Plot categorical features
    for col in categorical_cols:
        if plot_idx < len(axes):
            value_counts = df[col].value_counts()
            axes[plot_idx].bar(range(len(value_counts)), value_counts.values)
            axes[plot_idx].set_title(f"{col} (Categorical)")
            axes[plot_idx].set_xlabel(col)
            axes[plot_idx].set_ylabel("Count")
            axes[plot_idx].set_xticks(range(len(value_counts)))
            axes[plot_idx].set_xticklabels(value_counts.index, rotation=45)
            plot_idx += 1
    
    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle("Feature Distributions")
    plt.tight_layout()
    return fig

def plot_roc_curve(y_true, y_pred_proba, model_name="Model"):
    """ROC Curve for binary classification"""
    if len(np.unique(y_true)) != 2:
        return None
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name}')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
    """Confusion Matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    return fig

def plot_feature_importance(model, feature_names, model_name="Model"):
    """Feature Importance for tree-based models"""
    if not hasattr(model, 'feature_importances_'):
        return None
    
    importances = model.feature_importances_
    n_features = len(importances)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(n_features), importances)
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    ax.set_title(f'Feature Importance - {model_name}')
    ax.set_xticks(range(n_features))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_actual_vs_predicted(y_true, y_pred, model_name="Model"):
    """Actual vs Predicted for regression"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.6)
    
    # Perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'Actual vs Predicted - {model_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add R² score
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    return fig

# Preprocessing function
def preprocess_data(df, target_col):
    """Simple preprocessing"""
    df_processed = df.dropna()
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]
    
    # Encode categorical features
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y.astype(str))
    
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

# Main App Interface
st.sidebar.header("📂 Dataset Options")

# Dataset selection
dataset_option = st.sidebar.radio("Choose dataset:", ["Sample Data", "Upload CSV"])

df = None

if dataset_option == "Sample Data":
    if st.sidebar.button("Generate Sample Data"):
        df = create_sample_data()
        st.sidebar.success("✅ Sample data generated!")

elif dataset_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

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
        st.metric("Categorical Features", len(df.select_dtypes(include=['object']).columns))

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
            
            st.success(f"✅ Data preprocessed successfully: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
            
        except Exception as e:
            st.error(f"❌ Preprocessing failed: {str(e)}")
            st.stop()
        
        # Model Training
        try:
            with st.spinner("🤖 Training models..."):
                results, trained_models = train_and_evaluate(X_train, X_test, y_train, y_test)
                
        except Exception as e:
            st.error(f"❌ Model training failed: {str(e)}")
            st.stop()
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["📊 EDA", "🤖 Model Results", "📈 Predictions"])
        
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
                    # Actual vs Predicted for regression
                    actual_pred_plot = plot_actual_vs_predicted(
                        y_test, metrics['predictions'], model_name
                    )
                    if actual_pred_plot:
                        st.pyplot(actual_pred_plot)
                    
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
            
            # Manual Predictions
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

else:
    st.info("👆 Please select a dataset option from the sidebar to get started!")



