import os
import json
import time
import re
from typing import Dict, List, Any, Optional
from collections import Counter


import os
import json
import time
import re
from typing import Dict, List, Any, Optional
from collections import Counter
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class ReasoningAgent:
    """Agent that uses multiple inference-time techniques to solve reasoning problems."""
    
    def __init__(self, api_key: str = "cse476", 
                 api_base: str = "http://10.4.58.53:41701/v1",
                 model: str = "bens_model"):
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.call_count = 0
        self._lock = threading.Lock()  # Thread safety for call_count
        
    def call_llm(self, prompt: str, system: str = None, 
                 temperature: float = 0.0, max_tokens: int = 2048) -> Optional[str]:
        """Call the LLM API and return the response text."""
        with self._lock:
            self.call_count += 1
        
        url = f"{self.api_base}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                print(f"API Error: {resp.status_code}")
                return None
        except Exception as e:
            print(f"Exception calling LLM: {e}")
            return None