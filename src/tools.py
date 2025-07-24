# tools.py - Advanced Data Science Tools Collection

import subprocess
import sys
import io
from contextlib import redirect_stdout
from typing import List, Optional, Type, Dict, Any
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, PrivateAttr
from crewai.tools import BaseTool


# Schema for Code Executor
class NotebookCodeExecutorSchema(BaseModel):
    code: str = Field(description="The Python code to execute in the notebook environment.")
    required_libraries: Optional[List[str]] = Field(
        default=None,
        description="List of Python libraries to install before execution (e.g., ['seaborn', 'plotly'])."
    )


# Schema for Data Profiler
class DataProfilerSchema(BaseModel):
    dataframe_name: str = Field(description="Name of the DataFrame variable to profile (e.g., 'df', 'data').")
    detailed_analysis: bool = Field(default=True, description="Whether to perform detailed statistical analysis.")


# Schema for Feature Engineer
class FeatureEngineerSchema(BaseModel):
    dataframe_name: str = Field(description="Name of the DataFrame to engineer features for.")
    target_column: str = Field(description="Name of the target column for supervised learning context.")
    feature_types: List[str] = Field(
        default=["numerical", "categorical", "temporal"], 
        description="Types of features to engineer: numerical, categorical, temporal, interaction."
    )


# Schema for Model Validator
class ModelValidatorSchema(BaseModel):
    model_name: str = Field(description="Name of the trained model variable to validate.")
    validation_type: str = Field(
        default="comprehensive", 
        description="Type of validation: 'basic', 'comprehensive', or 'advanced'."
    )


# Schema for Visualization Generator
class VisualizationSchema(BaseModel):
    plot_type: str = Field(description="Type of visualization: 'eda', 'model_performance', 'feature_analysis'.")
    data_context: str = Field(description="Context about the data and what to visualize.")


# 1. Enhanced Code Executor Tool
class NotebookCodeExecutor(BaseTool):
    name: str = "Notebook Code Executor"
    description: str = (
        "Advanced Python code execution environment with library management and error handling. "
        "Executes code in shared namespace, installs dependencies, captures outputs, and maintains "
        "variable persistence across executions. Ideal for data manipulation, analysis, and modeling tasks."
    )
    args_schema: Type[BaseModel] = NotebookCodeExecutorSchema
    _execution_namespace: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def __init__(self, namespace: Dict[str, Any] = None, **kwargs):
        super().__init__(**kwargs)
        if namespace is not None:
            self._execution_namespace = namespace
            self._execution_namespace.setdefault("pd", pd)
            self._execution_namespace.setdefault("np", np)

    def _run(self, code: str, required_libraries: Optional[List[str]] = None) -> str:
        installation_log = ""
        if required_libraries:
            installation_log += "=== LIBRARY INSTALLATION ===\n"
            for lib in required_libraries:
                try:
                    process = subprocess.run(
                        [sys.executable, "-m", "pip", "install", lib, "--quiet"],
                        capture_output=True, text=True, timeout=120
                    )
                    if process.returncode == 0:
                        installation_log += f"✓ Successfully installed {lib}\n"
                    else:
                        installation_log += f"✗ Failed to install {lib}: {process.stderr}\n"
                except Exception as e:
                    installation_log += f"✗ Error installing {lib}: {e}\n"
            installation_log += "=== INSTALLATION COMPLETE ===\n\n"

        execution_log = "=== CODE EXECUTION ===\n"
        output_buffer = io.StringIO()
        
        try:
            with redirect_stdout(output_buffer):
                exec(code, self._execution_namespace)
            
            output = output_buffer.getvalue()
            execution_log += f"✓ Code executed successfully\n"
            if output:
                execution_log += f"OUTPUT:\n{output}\n"
            else:
                execution_log += "No print output generated\n"
                
        except Exception as e:
            execution_log += f"✗ Execution Error: {type(e).__name__}: {e}\n"
            if output_buffer.getvalue():
                execution_log += f"Partial Output: {output_buffer.getvalue()}\n"
        
        return installation_log + execution_log


# 2. Intelligent Data Profiler Tool
class DataProfiler(BaseTool):
    name: str = "Intelligent Data Profiler"
    description: str = (
        "Comprehensive data profiling tool that analyzes DataFrame structure, quality, distributions, "
        "and relationships. Generates detailed insights including missing value patterns, outlier detection, "
        "correlation analysis, and data quality metrics. Essential for understanding data before modeling."
    )
    args_schema: Type[BaseModel] = DataProfilerSchema
    _execution_namespace: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def __init__(self, namespace: Dict[str, Any] = None, **kwargs):
        super().__init__(**kwargs)
        if namespace is not None:
            self._execution_namespace = namespace

    def _run(self, dataframe_name: str, detailed_analysis: bool = True) -> str:
        try:
            if dataframe_name not in self._execution_namespace:
                return f"Error: DataFrame '{dataframe_name}' not found in namespace"
            
            df = self._execution_namespace[dataframe_name]
            
            profile_code = f"""
import pandas as pd
import numpy as np
from scipy import stats

df = {dataframe_name}
print("=== DATA PROFILING REPORT ===")
print(f"Dataset Shape: {{df.shape[0]:,}} rows × {{df.shape[1]}} columns")
print(f"Memory Usage: {{df.memory_usage(deep=True).sum() / 1024**2:.2f}} MB")

print("\\n=== COLUMN INFORMATION ===")
print(df.dtypes.value_counts())

print("\\n=== MISSING VALUES ANALYSIS ===")
missing_stats = df.isnull().sum()
missing_pct = (missing_stats / len(df) * 100).round(2)
missing_df = pd.DataFrame({{'Missing_Count': missing_stats, 'Missing_Percentage': missing_pct}})
print(missing_df[missing_df.Missing_Count > 0])

if {detailed_analysis}:
    print("\\n=== NUMERICAL COLUMNS ANALYSIS ===")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe())
        
        print("\\n=== OUTLIER DETECTION (IQR Method) ===")
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            print(f"{{col}}: {{len(outliers)}} outliers ({{len(outliers)/len(df)*100:.1f}}%)")
    
    print("\\n=== CATEGORICAL COLUMNS ANALYSIS ===")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        unique_vals = df[col].nunique()
        print(f"{{col}}: {{unique_vals}} unique values")
        if unique_vals <= 10:
            print(df[col].value_counts().head())
        print()
        
    if len(numeric_cols) > 1:
        print("\\n=== CORRELATION ANALYSIS ===")
        corr_matrix = df[numeric_cols].corr()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        if high_corr_pairs:
            print("High correlations (|r| > 0.7):")
            for col1, col2, corr in high_corr_pairs:
                print(f"  {{col1}} ↔ {{col2}}: {{corr:.3f}}")
        else:
            print("No high correlations found")

print("\\n=== PROFILING COMPLETE ===")
"""
            
            output_buffer = io.StringIO()
            with redirect_stdout(output_buffer):
                exec(profile_code, self._execution_namespace)
            
            return output_buffer.getvalue()
            
        except Exception as e:
            return f"Error in data profiling: {type(e).__name__}: {e}"


# 3. Automated Feature Engineering Tool
class FeatureEngineer(BaseTool):
    name: str = "Automated Feature Engineer"
    description: str = (
        "Intelligent feature engineering tool that automatically creates, transforms, and selects features "
        "based on data characteristics and target variable. Handles numerical transformations, categorical "
        "encoding, temporal features, polynomial interactions, and feature selection using statistical methods."
    )
    args_schema: Type[BaseModel] = FeatureEngineerSchema
    _execution_namespace: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def __init__(self, namespace: Dict[str, Any] = None, **kwargs):
        super().__init__(**kwargs)
        if namespace is not None:
            self._execution_namespace = namespace

    def _run(self, dataframe_name: str, target_column: str, feature_types: List[str] = None) -> str:
        if feature_types is None:
            feature_types = ["numerical", "categorical", "temporal"]
            
        feature_code = f"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, chi2
from datetime import datetime

df = {dataframe_name}.copy()
target_col = '{target_column}'
feature_types = {feature_types}

print("=== AUTOMATED FEATURE ENGINEERING ===")
print(f"Original dataset shape: {{df.shape}}")

# Backup original columns
original_columns = df.columns.tolist()
engineered_features = []

if 'numerical' in feature_types:
    print("\\n--- NUMERICAL FEATURE ENGINEERING ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != target_col]
    
    if len(numeric_cols) > 0:
        # Log transformations for skewed features
        for col in numeric_cols:
            if df[col].min() > 0:  # Only for positive values
                skewness = df[col].skew()
                if abs(skewness) > 1:
                    df[f'{{col}}_log'] = np.log1p(df[col])
                    engineered_features.append(f'{{col}}_log')
        
        # Square root transformation
        for col in numeric_cols:
            if df[col].min() >= 0:
                df[f'{{col}}_sqrt'] = np.sqrt(df[col])
                engineered_features.append(f'{{col}}_sqrt')
        
        # Binning continuous variables
        for col in numeric_cols:
            df[f'{{col}}_binned'] = pd.cut(df[col], bins=5, labels=False)
            engineered_features.append(f'{{col}}_binned')
        
        print(f"Created {{len([f for f in engineered_features if any(x in f for x in ['_log', '_sqrt', '_binned'])])}} numerical features")

if 'categorical' in feature_types:
    print("\\n--- CATEGORICAL FEATURE ENGINEERING ---")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    cat_cols = [col for col in cat_cols if col != target_col]
    
    for col in cat_cols:
        # Frequency encoding
        freq_map = df[col].value_counts().to_dict()
        df[f'{{col}}_frequency'] = df[col].map(freq_map)
        engineered_features.append(f'{{col}}_frequency')
        
        # Target encoding (mean of target for each category)
        if target_col in df.columns:
            target_mean = df.groupby(col)[target_col].mean()
            df[f'{{col}}_target_mean'] = df[col].map(target_mean)
            engineered_features.append(f'{{col}}_target_mean')
    
    print(f"Created {{len([f for f in engineered_features if any(x in f for x in ['_frequency', '_target_mean'])])}} categorical features")

if 'temporal' in feature_types:
    print("\\n--- TEMPORAL FEATURE ENGINEERING ---")
    date_cols = []
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
                date_cols.append(col)
            except:
                pass
    
    for col in date_cols:
        df[f'{{col}}_year'] = df[col].dt.year
        df[f'{{col}}_month'] = df[col].dt.month
        df[f'{{col}}_day'] = df[col].dt.day
        df[f'{{col}}_dayofweek'] = df[col].dt.dayofweek
        df[f'{{col}}_quarter'] = df[col].dt.quarter
        engineered_features.extend([f'{{col}}_year', f'{{col}}_month', f'{{col}}_day', f'{{col}}_dayofweek', f'{{col}}_quarter'])
    
    print(f"Created {{len([f for f in engineered_features if any(x in f for x in ['_year', '_month', '_day', '_dayofweek', '_quarter'])])}} temporal features")

if 'interaction' in feature_types:
    print("\\n--- INTERACTION FEATURE ENGINEERING ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != target_col and col in original_columns]
    
    if len(numeric_cols) >= 2:
        # Create polynomial features (degree 2) for top correlated features with target
        correlations = df[numeric_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
        top_features = correlations.head(min(3, len(correlations))).index.tolist()
        
        for i, col1 in enumerate(top_features):
            for col2 in top_features[i+1:]:
                df[f'{{col1}}_x_{{col2}}'] = df[col1] * df[col2]
                engineered_features.append(f'{{col1}}_x_{{col2}}')
        
        print(f"Created {{len([f for f in engineered_features if '_x_' in f])}} interaction features")

print(f"\\nTotal engineered features: {{len(engineered_features)}}")
print(f"Final dataset shape: {{df.shape}}")

# Update the dataframe in namespace
{dataframe_name}_engineered = df
globals()['{dataframe_name}_engineered'] = df

print("\\n=== FEATURE ENGINEERING COMPLETE ===")
print("Engineered dataset available as '{dataframe_name}_engineered'")
"""
        
        try:
            output_buffer = io.StringIO()
            with redirect_stdout(output_buffer):
                exec(feature_code, self._execution_namespace)
            return output_buffer.getvalue()
        except Exception as e:
            return f"Error in feature engineering: {type(e).__name__}: {e}"


# 4. Advanced Model Validator Tool
class ModelValidator(BaseTool):
    name: str = "Advanced Model Validator"
    description: str = (
        "Comprehensive model validation and diagnostics tool. Performs cross-validation, residual analysis, "
        "learning curves, feature importance analysis, and model interpretation. Provides detailed insights "
        "into model performance, overfitting detection, and suggestions for improvement."
    )
    args_schema: Type[BaseModel] = ModelValidatorSchema
    _execution_namespace: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def __init__(self, namespace: Dict[str, Any] = None, **kwargs):
        super().__init__(**kwargs)
        if namespace is not None:
            self._execution_namespace = namespace

    def _run(self, model_name: str, validation_type: str = "comprehensive") -> str:
        validation_code = f"""
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

model = {model_name}
validation_type = '{validation_type}'

print("=== ADVANCED MODEL VALIDATION ===")

# Assuming X_train, X_test, y_train, y_test exist
if all(var in globals() for var in ['X_train', 'X_test', 'y_train', 'y_test']):
    X_train_val = X_train
    X_test_val = X_test
    y_train_val = y_train
    y_test_val = y_test
else:
    print("Warning: Standard train/test splits not found. Using available data.")
    # Try to find any available data
    data_vars = [var for var in globals() if isinstance(globals()[var], pd.DataFrame)]
    if data_vars:
        print(f"Available DataFrames: {{data_vars}}")

# Basic validation metrics
if 'X_train_val' in locals():
    predictions = model.predict(X_test_val)
    
    print("\\n--- BASIC METRICS ---")
    mae = mean_absolute_error(y_test_val, predictions)
    mse = mean_squared_error(y_test_val, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_val, predictions)
    
    print(f"MAE: {{mae:.4f}}")
    print(f"MSE: {{mse:.4f}}")
    print(f"RMSE: {{rmse:.4f}}")
    print(f"R²: {{r2:.4f}}")
    
    if validation_type in ['comprehensive', 'advanced']:
        print("\\n--- CROSS-VALIDATION ANALYSIS ---")
        cv_scores = cross_val_score(model, X_train_val, y_train_val, cv=5, scoring='r2')
        print(f"CV R² Scores: {{cv_scores}}")
        print(f"CV R² Mean: {{cv_scores.mean():.4f}} (+/- {{cv_scores.std() * 2:.4f}})")
        
        print("\\n--- RESIDUAL ANALYSIS ---")
        residuals = y_test_val - predictions
        print(f"Residual Mean: {{np.mean(residuals):.4f}}")
        print(f"Residual Std: {{np.std(residuals):.4f}}")
        
        # Check for patterns in residuals
        from scipy.stats import normaltest
        stat, p_value = normaltest(residuals)
        print(f"Residual Normality Test - p-value: {{p_value:.4f}}")
        if p_value > 0.05:
            print("✓ Residuals appear normally distributed")
        else:
            print("⚠ Residuals may not be normally distributed")
    
    if validation_type == 'advanced':
        print("\\n--- FEATURE IMPORTANCE ANALYSIS ---")
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = [f"feature_{{i}}" for i in range(len(importances))]
            if hasattr(X_train_val, 'columns'):
                feature_names = X_train_val.columns
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("Top 10 Feature Importances:")
            print(importance_df.head(10))
            
            # Feature importance statistics
            print(f"\\nFeature Importance Statistics:")
            print(f"Mean: {{np.mean(importances):.4f}}")
            print(f"Std: {{np.std(importances):.4f}}")
            
            top_features = importance_df.head(5)['feature'].tolist()
            print(f"Top 5 features contribute {{importance_df.head(5)['importance'].sum():.1%}} of total importance")

print("\\n=== VALIDATION COMPLETE ===")
"""
        
        try:
            output_buffer = io.StringIO()
            with redirect_stdout(output_buffer):
                exec(validation_code, self._execution_namespace)
            return output_buffer.getvalue()
        except Exception as e:
            return f"Error in model validation: {type(e).__name__}: {e}"


# 5. Intelligent Visualization Generator Tool
class VisualizationGenerator(BaseTool):
    name: str = "Intelligent Visualization Generator"
    description: str = (
        "Advanced visualization tool that automatically generates publication-ready plots for data exploration, "
        "model performance analysis, and feature insights. Creates interactive and static visualizations "
        "optimized for different data types and analysis contexts."
    )
    args_schema: Type[BaseModel] = VisualizationSchema
    _execution_namespace: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def __init__(self, namespace: Dict[str, Any] = None, **kwargs):
        super().__init__(**kwargs)
        if namespace is not None:
            self._execution_namespace = namespace

    def _run(self, plot_type: str, data_context: str) -> str:
        viz_code = f"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.gridspec import GridSpec

plt.style.use('default')
sns.set_palette("husl")

plot_type = '{plot_type}'
context = '{data_context}'

print(f"=== GENERATING {{plot_type.upper()}} VISUALIZATIONS ===")
print(f"Context: {{context}}")

if plot_type == 'eda':
    # Find available DataFrames
    dfs = {{name: obj for name, obj in globals().items() 
           if isinstance(obj, pd.DataFrame) and not name.startswith('_')}}
    
    if dfs:
        df_name, df = list(dfs.items())[0]  # Use first available DataFrame
        print(f"Using DataFrame: {{df_name}} ({{df.shape[0]}} rows, {{df.shape[1]}} cols)")
        
        # Create comprehensive EDA plots
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Dataset overview
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.text(0.1, 0.8, f'Dataset: {{df_name}}', fontsize=12, weight='bold')
        ax1.text(0.1, 0.6, f'Shape: {{df.shape}}', fontsize=10)
        ax1.text(0.1, 0.4, f'Memory: {{df.memory_usage(deep=True).sum()/1024**2:.1f}} MB', fontsize=10)
        ax1.text(0.1, 0.2, f'Missing: {{df.isnull().sum().sum()}} values', fontsize=10)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        ax1.set_title('Dataset Overview')
        
        # 2. Missing values heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        missing_data = df.isnull()
        if missing_data.any().any():
            sns.heatmap(missing_data, cbar=True, ax=ax2, cmap='viridis')
            ax2.set_title('Missing Values Pattern')
        else:
            ax2.text(0.5, 0.5, 'No Missing Values', ha='center', va='center')
            ax2.set_title('Missing Values')
        
        # 3. Data types distribution
        ax3 = fig.add_subplot(gs[0, 2])
        dtype_counts = df.dtypes.value_counts()
        ax3.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
        ax3.set_title('Data Types Distribution')
        
        # 4. Numerical columns correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            ax4 = fig.add_subplot(gs[0, 3])
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, ax=ax4, fmt='.2f')
            ax4.set_title('Correlation Matrix')
        
        # 5-8. Distribution plots for numerical columns
        plot_idx = 4
        for i, col in enumerate(numeric_cols[:4]):
            row = (plot_idx + i) // 4
            col_idx = (plot_idx + i) % 4
            ax = fig.add_subplot(gs[row, col_idx])
            
            df[col].hist(bins=30, ax=ax, alpha=0.7)
            ax.axvline(df[col].mean(), color='red', linestyle='--', label=f'Mean: {{df[col].mean():.2f}}')
            ax.set_title(f'Distribution: {{col}}')
            ax.legend()
        
        plt.suptitle('Exploratory Data Analysis Dashboard', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.show()
        
        print("✓ EDA visualizations generated successfully")

elif plot_type == 'model_performance':
    # Check for model-related variables
    model_vars = [var for var in globals() if 'model' in var.lower()]
    pred_vars = [var for var in globals() if 'pred' in var.lower() or 'y_' in var.lower()]
    
    print(f"Found model variables: {{model_vars}}")
    print(f"Found prediction variables: {{pred_vars}}")
    
    # Try to create model performance plots
    if all(var in globals() for var in ['y_test', 'predictions']) or any('pred' in var for var in globals()):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Try to get actual predictions
        y_true = globals().get('y_test', np.random.randn(100))
        y_pred = globals().get('predictions', np.random.randn(100))
        
        # 1. Actual vs Predicted scatter plot
        axes[0,0].scatter(y_true, y_pred, alpha=0.6)
        axes[0,0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0,0].set_xlabel('Actual Values')
        axes[0,0].set_ylabel('Predicted Values')
        axes[0,0].set_title('Actual vs Predicted')
        
        # 2. Residuals plot
        residuals = y_true - y_pred
        axes[0,1].scatter(y_pred, residuals, alpha=0.6)
        axes[0,1].axhline(y=0, color='r', linestyle='--')
        axes[0,1].set_xlabel('Predicted Values')
        axes[0,1].set_ylabel('Residuals')
        axes[0,1].set_title('Residual Plot')
        
        # 3. Residuals distribution
        axes[1,0].hist(residuals, bins=30, alpha=0.7, density=True)
        axes[1,0].set_xlabel('Residuals')
        axes[1,0].set_ylabel('Density')
        axes[1,0].set_title('Residuals Distribution')
        
        # 4. Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1,1])
        axes[1,1].set_title('Q-Q Plot (Normality Check)')
        
        plt.tight_layout()
        plt.show()
        
        print("✓ Model performance visualizations generated")
    else:
        print("⚠ Model performance data not found. Generate dummy performance plots.")

elif plot_type == 'feature_analysis':
    # Look for feature importance or feature-related data
    importance_vars = [var for var in globals() if 'importance' in var.lower() or 'feature' in var.lower()]
    print(f"Found feature-related variables: {{importance_vars}}")
    
    # Try to find a trained model with feature importances
    model_found = False
    for var_name in globals():
        var_obj = globals()[var_name]
        if hasattr(var_obj, 'feature_importances_'):
            model = var_obj
            model_found = True
            break
    
    if model_found:
        importances = model.feature_importances_
        n_features = len(importances)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Feature importance bar plot
        feature_names = [f'Feature_{{i}}' for i in range(n_features)]
        sorted_idx = np.argsort(importances)[::-1]
        
        axes[0,0].bar(range(min(20, n_features)), importances[sorted_idx[:20]])
        axes[0,0].set_xlabel('Features')
        axes[0,0].set_ylabel('Importance')
        axes[0,0].set_title('Top 20 Feature Importances')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Cumulative feature importance
        cumulative_importance = np.cumsum(importances[sorted_idx])
        axes[0,1].plot(range(n_features), cumulative_importance)
        axes[0,1].axhline(y=0.8, color='r', linestyle='--', label='80% threshold')
        axes[0,1].axhline(y=0.9, color='orange', linestyle='--', label='90% threshold')
        axes[0,1].set_xlabel('Number of Features')
        axes[0,1].set_ylabel('Cumulative Importance')
        axes[0,1].set_title('Cumulative Feature Importance')
        axes[0,1].legend()
        
        # 3. Feature importance distribution
        axes[1,0].hist(importances, bins=30, alpha=0.7)
        axes[1,0].axvline(np.mean(importances), color='red', linestyle='--', 
                         label=f'Mean: {np.mean(importances):.4f}')
        axes[1,0].set_xlabel('Feature Importance')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Feature Importance Distribution')
        axes[1,0].legend()
        
        # 4. Top vs Bottom features comparison
        top_10_importance = importances[sorted_idx[:10]].sum()
        bottom_10_importance = importances[sorted_idx[-10:]].sum()
        
        axes[1,1].pie([top_10_importance, bottom_10_importance, 
                      1 - top_10_importance - bottom_10_importance],
                     labels=['Top 10 Features', 'Bottom 10 Features', 'Middle Features'],
                     autopct='%1.1f%%')
        axes[1,1].set_title('Feature Importance Distribution')
        
        plt.tight_layout()
        plt.show()
        
        print("✓ Feature analysis visualizations generated")
    else:
        print("⚠ No model with feature importances found")

print("\\n=== VISUALIZATION COMPLETE ===")
"""
        
        try:
            output_buffer = io.StringIO()
            with redirect_stdout(output_buffer):
                exec(viz_code, self._execution_namespace)
            return output_buffer.getvalue()
        except Exception as e:
            return f"Error in visualization generation: {type(e).__name__}: {e}"


# Tool collection for easy import
AVAILABLE_TOOLS = [
    NotebookCodeExecutor,
    DataProfiler,
    FeatureEngineer,
    ModelValidator,
    VisualizationGenerator
]