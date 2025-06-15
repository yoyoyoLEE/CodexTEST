import streamlit as st
import os
import pandas as pd
import numpy as np
import httpx
from dotenv import load_dotenv
from sklearn.impute import SimpleImputer, KNNImputer
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

            # Automatic analysis on upload
            if 'analysis_done' not in st.session_state or st.session_state.uploaded_file != uploaded_file.name:
                with st.spinner("Analyzing data with AI..."):
                    import asyncio
                    analysis = asyncio.run(llm_client.analyze_data(df))
                    if analysis:
                        st.session_state.analysis = analysis["choices"][0]["message"]["content"]
                        st.session_state.analysis_done = True
                        st.session_state.uploaded_file = uploaded_file.name
            
            if 'analysis' in st.session_state:
                st.subheader("AI Cleaning Suggestions")
                st.write(st.session_state.analysis)
                
                if st.button("Apply Suggestions"):
                    with st.spinner("Generating and executing cleaning code..."):
                        # Generate cleaning code based on analysis
                        prompt = f"""
                        Based on these data cleaning suggestions:
                        {st.session_state.analysis}
                        
                        Generate clean, executable Python code using pandas to implement these suggestions.
                        The code must:
                        1. Start with all required imports (pandas as pd, numpy as np)
                        2. Operate on a dataframe variable called 'df'
                        3. Be properly formatted with correct Python syntax
                        4. Include exactly one complete code block
                        5. Have no markdown, explanations or comments
                        6. Handle all edge cases mentioned in the suggestions
                        7. Use errors='coerce' when converting to numeric
                        8. Handle string values carefully when converting to numbers
                        
                        Example of valid output:
                        import pandas as pd
                        import numpy as np
                        # Clean string columns first
                        df['duration'] = df['duration'].str.extract('(\d+)')[0]
                        # Then convert to numeric
                        df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
                        # Handle missing values
                        df = df.fillna(df.mean())
                        """
                        
                        headers = {
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json"
                        }

                        payload = {
                            "model": "deepseek/deepseek-chat-v3-0324:free",
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.3,
                            "max_tokens": 1000
                        }

                        try:
                            with httpx.Client() as client:
                                response = client.post(
                                    f"{llm_client.base_url}/chat/completions",
                                    headers=headers,
                                    json=payload,
                                    timeout=30.0
                                )
                                response.raise_for_status()
                                raw_response = response.json()["choices"][0]["message"]["content"]
                                
                                # Extract just the code block from the response
                                code = ""
                                in_code_block = False
                                for line in raw_response.split('\n'):
                                    if line.strip().startswith('```python'):
                                        in_code_block = True
                                        continue
                                    elif line.strip() == '```':
                                        in_code_block = False
                                        continue
                                    if in_code_block or (line.strip() and not line.strip().startswith('#')):
                                        code += line + '\n'
                                
                                # Validate and execute the generated code
                                try:
                                    # First check syntax
                                    compile(code, '<string>', 'exec')
                                    if not code.strip():
                                        raise ValueError("No executable code was generated")
                                    
                                    # Then execute in a safe environment with proper imports
                                    local_vars = {'df': df.copy()}
                                    global_vars = {
                                        'pd': pd,
                                        'np': np,
                                        'basic_imputation': basic_imputation,
                                        'groupwise_imputation': groupwise_imputation,
                                        'knn_imputation': knn_imputation,
                                        'mice_imputation': mice_imputation,
                                        'SimpleImputer': SimpleImputer,
                                        'KNNImputer': KNNImputer
                                    }
                                    exec(code, global_vars, local_vars)
                                    cleaned_df = local_vars['df']
                                    
                                    st.success("Suggestions applied successfully!")
                                    st.write("Cleaned Data:")
                                    st.dataframe(cleaned_df)
                                    st.download_button(
                                        "Download Cleaned Data",
                                        cleaned_df.to_csv(index=False),
                                        "cleaned_data.csv"
                                    )
                                    
                                    st.subheader("Applied Cleaning Code:")
                                    st.code(code, language='python')
                                    
                                except SyntaxError as e:
                                    st.error(f"Invalid code generated:\n{e}")
                                    st.code(code, language='python')
                                    st.warning("Please try analyzing the data again or provide more specific instructions")
                                except Exception as e:
                                    st.error(f"Error executing cleaning code:\n{e}")
                                    st.code(code, language='python')
                                    st.warning("The generated code may need manual adjustment")
                                    
                        except httpx.HTTPStatusError as e:
                            st.error(f"API request failed: {e.response.status_code} {e.response.reason_phrase}")
                        except Exception as e:
                            st.error(f"Error generating cleaning code: {str(e)}")
    
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
