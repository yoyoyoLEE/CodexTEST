import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv
from data_cleaning import (
    basic_imputation,
    groupwise_imputation,
    knn_imputation,
    mice_imputation,
    auto_impute
)
from llm_helpers import OpenRouterClient

# Load environment variables from .env file
load_dotenv()

def get_api_key():
    """Get API key from Streamlit secrets or .env file"""
    try:
        # Try to get from Streamlit secrets (production)
        return st.secrets["openrouter"]["api_key"]
    except:
        try:
            # Fallback to .env file (development)
            return os.getenv("OPENROUTER_API_KEY")
        except:
            return None

# Initialize the app
st.title("AI Data Cleaning Assistant")
st.write("Powered by OpenRouter DeepSeek model")

# Get API key
api_key = get_api_key()

if api_key:
    # Initialize LLM client
    llm_client = OpenRouterClient()
    
    # Create tabs
    tab1, tab2 = st.tabs(["Data Cleaning", "AI Assistant"])
    
    with tab1:
        # Data cleaning interface
        st.header("Data Cleaning Tools")
        uploaded_file = st.file_uploader("Upload data file", type=["csv", "xls", "xlsx"])
        
        if uploaded_file:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:  # Excel file
                df = pd.read_excel(uploaded_file)
            
            st.write("Original Data:")
            st.dataframe(df)
            
            st.write("Missing Values Summary:")
            st.dataframe(df.isnull().sum().rename("Missing Values"))
            
            method = st.selectbox(
                "Select imputation method:",
                ["Basic (mean/mode)", "Group-wise", "KNN", "MICE", "Auto"]
            )
            
            if st.button("Clean Data"):
                with st.spinner("Cleaning data..."):
                    if method == "Basic (mean/mode)":
                        cleaned_df = basic_imputation(df.copy())
                    elif method == "Group-wise":
                        group_col = st.selectbox("Select grouping column:", df.columns)
                        cleaned_df = groupwise_imputation(df.copy(), group_col)
                    elif method == "KNN":
                        cleaned_df = knn_imputation(df.copy())
                    elif method == "MICE":
                        cleaned_df = mice_imputation(df.copy())
                    else:
                        cleaned_df = auto_impute(df.copy())
                        
                    st.write("Cleaned Data:")
                    st.dataframe(cleaned_df)
                    st.download_button(
                        "Download Cleaned Data",
                        cleaned_df.to_csv(index=False),
                        "cleaned_data.csv"
                    )
    
    with tab2:
        # AI Assistant interface
        st.header("AI Data Cleaning Assistant")
        if 'messages' not in st.session_state:
            st.session_state.messages = []
            
        if uploaded_file:
            # Show data analysis
            if st.button("Analyze Data"):
                with st.spinner("Analyzing data..."):
                    import asyncio
                    analysis = asyncio.run(llm_client.analyze_data(df))
                    if analysis:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": analysis["choices"][0]["message"]["content"]
                        })
            
            # Display chat messages
            for msg in st.session_state.messages:
                st.chat_message(msg["role"]).write(msg["content"])
                
            # Chat input
            if prompt := st.chat_input("Ask about your data..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.chat_message("user").write(prompt)
                
                with st.spinner("Thinking..."):
                    response = llm_client.get_chat_response(st.session_state.messages)
                    if response:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response
                        })
                        st.chat_message("assistant").write(response)
        else:
            st.info("Please upload a CSV file in the Data Cleaning tab first")
else:
    st.error("""
        API key not found. Please configure:
        1. For local development: Create .env file with OPENROUTER_API_KEY
        2. For production: Add to Streamlit secrets.toml under [openrouter] section
    """)
