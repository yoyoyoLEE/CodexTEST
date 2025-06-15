import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_api_key():
    """Get API key from Streamlit secrets or .env file"""
    try:
        # Try to get from Streamlit secrets (production)
        return st.secrets["cline"]["api_key"]
    except:
        try:
            # Fallback to .env file (development)
            return os.getenv("CLINE_API_KEY")
        except:
            return None

# Initialize the app
st.title("CLINE API Web Interface")
st.write("Secure API key handling demonstration")

# Get API key
api_key = get_api_key()

if api_key:
    st.success("API key loaded successfully!")
    st.code(f"API key: {api_key[:4]}...{api_key[-4:]}", language="python")
    
    # Here you would add your actual API integration
    # Example:
    # response = your_api_call(api_key, ...)
    # st.write(response)
else:
    st.error("""
        API key not found. Please configure:
        1. For local development: Create .env file with CLINE_API_KEY
        2. For production: Add to Streamlit secrets.toml
    """)
