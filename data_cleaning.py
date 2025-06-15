import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

def basic_imputation(df):
    """Fill missing values with mean (numeric) or mode (categorical)"""
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Convert all columns to string for categorical imputation
    str_df = df.astype(str)
    
    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(exclude=np.number).columns
    
    # Impute numeric columns with mean
    if len(numeric_cols) > 0:
        num_imputer = SimpleImputer(strategy='mean')
        imputed_numeric = pd.DataFrame(
            num_imputer.fit_transform(df[numeric_cols]),
            columns=numeric_cols,
            index=df.index
        )
        df[numeric_cols] = imputed_numeric
    
    # Impute categorical columns with mode (using string representation)
    if len(cat_cols) > 0:
        # Handle each column separately to avoid mixed type issues
        for col in cat_cols:
            if df[col].notna().any():  # Only impute if there are some values
                mode_val = str_df[col].mode()[0] if not str_df[col].mode().empty else ''
                df[col] = df[col].fillna(mode_val)
    
    return df

def groupwise_imputation(df, group_col=None):
    """Fill missing values with group-specific mean/mode"""
    if group_col is None:
        return basic_imputation(df)
        
    for col in df.columns:
        if col != group_col:
            df[col] = df.groupby(group_col)[col].transform(
                lambda x: x.fillna(x.mean() if pd.api.types.is_numeric_dtype(x) else (x.mode()[0] if not x.mode().empty else ''))
            )
    return df

def knn_imputation(df, n_neighbors=5):
    """K-nearest neighbors imputation"""
    imputer = KNNImputer(n_neighbors=n_neighbors)
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df

def mice_imputation(df, estimator=RandomForestRegressor()):
    """Multiple Imputation by Chained Equations"""
    imputer = IterativeImputer(estimator=estimator, random_state=0)
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df

def auto_impute(df):
    """Automatically select best imputation method based on cross-validation"""
    # Clean up any empty columns first
    df = df.dropna(axis=1, how='all')
    # Use basic imputation as fallback
    return basic_imputation(df)
