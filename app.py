import streamlit as st
import pandas as pd
import asyncio
from data_cleaning import (
    basic_imputation,
    groupwise_imputation,
    knn_imputation,
    mice_imputation,
    auto_impute
)
from llm_helpers import OpenRouterClient

def main():
    st.title("AI-Powered Data Cleaning App")
    st.write("Upload a CSV or Excel file to clean missing data with LLM assistance")

    # Initialize OpenRouter client
    llm_client = OpenRouterClient()

    # Initialize session state for chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # File upload section
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
    
    if uploaded_file:
        # Read file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Store dataframe in session state
        st.session_state.df = df

        # Show original data
        with st.expander("Original Data"):
            st.write(df)

        # Show missing values
        st.subheader("Missing Values Summary")
        missing = df.isnull().sum()
        st.write(missing[missing > 0])

        # LLM Analysis Section
        st.subheader("AI Data Analysis")
        if st.button("Get Cleaning Recommendations"):
            with st.spinner("Analyzing data with AI..."):
                analysis = asyncio.run(llm_client.analyze_data(df))
                if analysis:
                    st.session_state.analysis = analysis
                    st.write(analysis['choices'][0]['message']['content'])
                else:
                    st.error("Failed to get analysis from LLM")

        # Chat Interface
        st.subheader("Data Cleaning Assistant")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask about data cleaning..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response = llm_client.get_chat_response(st.session_state.messages)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

        # Imputation method selection
        st.subheader("Cleaning Options")
        method = st.selectbox(
            "Choose imputation method:",
            ["Auto Select", "Basic (mean/mode)", "Group-wise", "KNN", "MICE"]
        )

        # Process data
        if st.button("Clean Data"):
            if "df" not in st.session_state:
                st.error("Please upload a file first")
                return

            with st.spinner("Processing..."):
                df = st.session_state.df
                if method == "Auto Select":
                    cleaned_df = auto_impute(df)
                elif method == "Basic (mean/mode)":
                    cleaned_df = basic_imputation(df)
                elif method == "Group-wise":
                    cleaned_df = groupwise_imputation(df)
                elif method == "KNN":
                    cleaned_df = knn_imputation(df)
                elif method == "MICE":
                    cleaned_df = mice_imputation(df)

                st.subheader("Cleaned Data")
                st.write(cleaned_df)

                # Show missing values after
                st.subheader("Missing Values After Cleaning")
                st.write(cleaned_df.isnull().sum())

                # Download cleaned file
                st.download_button(
                    label="Download Cleaned Data",
                    data=cleaned_df.to_csv(index=False).encode('utf-8'),
                    file_name='cleaned_data.csv',
                    mime='text/csv'
                )

if __name__ == "__main__":
    main()
