import httpx
import json
from typing import Optional, Dict, Any
import pandas as pd
import streamlit as st

class OpenRouterClient:
    def __init__(self):
        self.base_url = "https://openrouter.ai/api/v1"
        try:
            # Try to get from Streamlit secrets
            self.api_key = st.secrets["openrouter"]["api_key"]
        except:
            try:
                # Fallback to .env file
                from dotenv import load_dotenv
                import os
                load_dotenv()
                self.api_key = os.getenv("OPENROUTER_API_KEY")
            except:
                self.api_key = None

    async def analyze_data(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if not self.api_key:
            return None

        # Prepare data summary for LLM (convert Timestamps to strings)
        def convert_timestamps(obj):
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_timestamps(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_timestamps(x) for x in obj]
            return obj

        data_summary = {
            "columns": list(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "sample_data": convert_timestamps(df.head().to_dict(orient="list"))
        }

        prompt = f"""Analyze this dataset and suggest the best data cleaning approach:
{json.dumps(convert_timestamps(data_summary), indent=2)}

Consider:
1. Types of missing data (MCAR, MAR, MNAR)
2. Column data types
3. Distribution of missing values
4. Relationships between columns

Provide:
- Recommended imputation methods for each column
- Justification for each recommendation
- Any warnings about data quality issues"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "deepseek/deepseek-chat-v3-0324:free",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1000
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                st.error(f"LLM API error: {e.response.status_code} {e.response.reason_phrase} for url {e.request.url}\nFor more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/{e.response.status_code}")
                if e.response.status_code == 400:
                    st.error(f"Request payload: {json.dumps(payload, indent=2)}")
                return None
            except Exception as e:
                st.error(f"LLM API error: {str(e)}")
                return None

    def get_chat_response(self, messages: list) -> Optional[str]:
        """Get response from chat interface"""
        if not self.api_key:
            return None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "deepseek/deepseek-chat-v3-0324:free",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000
        }

        with httpx.Client() as client:
            try:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                st.error(f"Chat API error: {str(e)}")
                return None
