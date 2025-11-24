# utils/llm_client.py

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    import requests

from typing import Dict, List, Any

class LLMClient:
    """LLM client, prioritizes official SDK, falls back to HTTP requests"""
    
    def __init__(self, api_key: str = None, base_url: str = "https://api.openai.com/v1", model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        
        if OPENAI_AVAILABLE and api_key:
            # Use official SDK
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            self.use_sdk = True
        else:
            # Fall back to HTTP requests
            self.client = None
            self.use_sdk = False
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}" if api_key else ""
            }
    
    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """Call LLM chat completion API"""
        
        if self.use_sdk:
            return self._chat_completion_sdk(messages, temperature, max_tokens)
        else:
            return self._chat_completion_http(messages, temperature, max_tokens)
    
    def _chat_completion_sdk(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
        """Call using official SDK"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI SDK call failed: {e}")
            return ""
    
    def _chat_completion_http(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
        """Call using HTTP requests"""
        url = f"{self.base_url}/chat/completions"
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            print(f"HTTP API call failed: {e}")
            return ""