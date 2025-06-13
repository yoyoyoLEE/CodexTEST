import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
import os
from sklearn.impute import SimpleImputer, KNNImputer

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def get_ai_suggestions(df):
    """Get AI suggestions for data cleaning"""
    sample_data = df.head().to_string()
    prompt = f"""Analyze this dataset sample and suggest cleaning steps:
    {sample_data}
    
    Provide:
    1. Data quality issues found
    2. Recommended cleaning operations
    3. Suggested imputation methods for missing values"""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

def handle_missing_values(df, strategy='mean'):
    """Handle missing values based on selected strategy"""
    if strategy == 'mean':
        imputer = SimpleImputer(strategy='mean')
    elif strategy == 'median':
        imputer = SimpleImputer(strategy='median')
    elif strategy == 'knn':
        imputer = KNNImputer(n_neighbors=5)
    else:  # constant
        imputer = SimpleImputer(strategy='constant', fill_value=0)
    
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df

def main():
    st.title("Ultimate Data Cleaner with AI Assistance")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your dataset", 
        type=['csv', 'xlsx', 'parquet', 'json', 'feather']
    )
    
    if uploaded_file:
        try:
            # Read file based on extension
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.parquet'):
                df = pd.read_parquet(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            elif uploaded_file.name.endswith('.feather'):
                df = pd.read_feather(uploaded_file)
                
            st.success("File successfully loaded!")
            
            # Data preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # AI suggestions
            if st.button("Get AI Cleaning Suggestions"):
                with st.spinner("Analyzing data with AI..."):
                    suggestions = get_ai_suggestions(df)
                    st.subheader("AI Suggestions")
                    st.markdown(suggestions)
            
            # Missing value handling
            st.subheader("Missing Value Handling")
            strategy = st.selectbox(
                "Select imputation strategy",
                ['mean', 'median', 'knn', 'constant']
            )
            
            if st.button("Preview Changes"):
                cleaned_df = handle_missing_values(df.copy(), strategy)
                st.subheader("Preview of Cleaned Data")
                st.dataframe(cleaned_df.head())
                
                if st.button("Apply Changes"):
                    df = cleaned_df
                    st.success("Changes applied to dataset!")
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
